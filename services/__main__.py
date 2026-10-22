"""python -m services - Luna's backend in rootless Podman containers.

    python -m services setup          everything it needs, asking first
    python -m services setup tts      her voice in a container
    python -m services setup --server --allow IP   this machine as a GPU server (deploy/)
    python -m services up [ROLE...]   start (or keep) chat, embed, rerank (and tts)
    python -m services down [ROLE...] stop and remove them; models stay
    python -m services status         each one, and whether it answers
    python -m services logs ROLE      a container's log (-f follows it)
    python -m services bench          where the side models go on a small GPU
    python -m services reset          remove what it made (asks; your models only if you say)

Roles: chat, embed, rerank, tts. Settings: llama/server.env (as for
llama/start.sh), then services/services.env.
"""
import argparse
import os
import sys

from services import CONTAINER, ROLES, keys, lifecycle, settings


def _roles(names, default, allowed=None):
    """The roles asked for (the default ones when none were), or exit."""
    allowed = allowed or default
    bad = [n for n in names if n not in allowed]
    if bad:
        sys.exit(f"Unknown role: {', '.join(bad)} (roles: {', '.join(allowed)})")
    return tuple(names) or default


def cmd_up(args):
    """Start or keep the roles, then wait for them to answer."""
    problem = lifecycle.podman_problem()
    if problem:
        sys.exit(problem)
    try:
        s = settings.load()
    except settings.SettingsError as e:
        sys.exit(str(e))
    warning = settings.host_warning(s)
    if warning:
        print("  ! " + warning)
        if not args.yes and sys.stdin.isatty():
            try:
                if input("  Go on? [y/N] ").strip().lower()[:1] != "y":
                    sys.exit(1)
            except EOFError:
                sys.exit(1)
    token, _ = keys.token_source()
    results = lifecycle.up(s, _roles(args.roles, lifecycle.default_roles(), ROLES), token)
    for r in results:
        mark = {"kept": "✓", "started": "✓", "skipped": "·", "failed": "✗"}[r.state]
        print(f"  {mark} {r.role}: {r.state}" + (f" - {r.note}" if r.note else ""))
    waiting = [r.role for r in results if r.state in ("kept", "started")]
    if waiting and not args.no_wait:
        timeout = int(s["READY_TIMEOUT"] or 600)
        for role in waiting:
            url = f"http://127.0.0.1:{settings.port(role, s)}"
            print(f"  waiting for {role} on {url}", end="", flush=True)
            try:
                ready = lifecycle.wait_ready(url, timeout, alive=lifecycle.running(role))
            except KeyboardInterrupt:
                print(" - not waiting (it carries on loading)")
                break
            if ready:
                print(" ready.")
                continue
            print(" not ready." if lifecycle.running(role)() else " it stopped.")
            hint = lifecycle.trouble(role)
            print(f"    {hint}" if hint else f"    python -m services logs {role} says why")
    return 1 if any(r.state == "failed" for r in results) else 0


def cmd_down(args):
    """Stop and remove ours."""
    gone = lifecycle.down(_roles(args.roles, tuple(CONTAINER)))
    print("  stopped: " + ", ".join(gone) if gone else "  nothing of ours was running")
    return 0


def cmd_status(args):
    """A line per role; exit 1 unless all answer."""
    s = settings.load()
    good = True
    for fine, line in lifecycle.status(s):
        good = good and fine
        print(("  ✓ " if fine else "  · ") + line)
    print(f"  key: llama/.api_key {'present' if keys.read_key() else 'missing - python -m services setup'}")
    return 0 if good else 1


def cmd_logs(args):
    """Hand over to podman logs."""
    os.execvp("podman", ["podman", "logs", *(["-f"] if args.follow else []), CONTAINER[args.role]])


def cmd_bench(args):
    """Where the side models should go on this GPU (services/bench.py)."""
    from services.bench import bench
    from services.setup import Asker

    return 0 if bench(Asker(args.yes)) else 1


def cmd_reset(args):
    """Remove what python -m services made; luna-models only if asked."""
    from services.setup import Asker

    gone = lifecycle.reset(Asker(args.yes).ask)
    print("  removed: " + ", ".join(gone) if gone else "  nothing removed")
    return 0


def cmd_setup(args):
    """The setup steps (services/setup.py)."""
    from services.setup import Asker, setup, setup_tts

    if args.server:
        from services.setup import setup_server

        return 0 if setup_server(Asker(args.yes), args.allow) else 1
    if args.what == "tts":
        return 0 if setup_tts(Asker(args.yes)) else 1
    return 0 if setup(Asker(args.yes), gpu=args.gpu) else 1


def main(argv=None):
    """Parse the command line and run the subcommand."""
    ap = argparse.ArgumentParser(prog="python -m services", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup", help="set everything up, asking before each step")
    p.add_argument("what", nargs="?", choices=("tts",), help="tts: her voice only")
    p.add_argument("--yes", action="store_true", help="take every default without asking")
    p.add_argument("--gpu", default="auto", help="auto (default), cpu, cuda, cuda13, rocm or vulkan")
    p.add_argument("--server", action="store_true", help="this machine serves another on the network (deploy/)")
    p.add_argument("--allow", metavar="IP", help="with --server: the one client to let through the firewall")
    p.set_defaults(func=cmd_setup)
    p = sub.add_parser("up", help="start or keep the containers")
    p.add_argument("roles", nargs="*")
    p.add_argument("--no-wait", action="store_true", help="don't wait for them to answer")
    p.add_argument("--yes", action="store_true", help="don't ask about a shared HOST")
    p.set_defaults(func=cmd_up)
    p = sub.add_parser("down", help="stop and remove them (models and secrets stay)")
    p.add_argument("roles", nargs="*")
    p.set_defaults(func=cmd_down)
    p = sub.add_parser("status", help="what's running, and answering")
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("bench", help="where the embedder and reranker should run on this GPU")
    p.add_argument("--yes", action="store_true", help="write the recommendation without asking")
    p.set_defaults(func=cmd_bench)
    p = sub.add_parser("reset", help="remove our containers, images, volumes and secrets (asks first)")
    p.add_argument("--yes", action="store_true", help="remove ours without asking (luna-models and other luna-* things stay)")
    p.set_defaults(func=cmd_reset)
    p = sub.add_parser("logs", help="a container's log")
    p.add_argument("role", choices=tuple(CONTAINER))
    p.add_argument("-f", "--follow", action="store_true")
    p.set_defaults(func=cmd_logs)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
