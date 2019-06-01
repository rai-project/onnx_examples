import click
import sys
import copy
import traceback

DEBUG = False
VERBOSE = False
QUIET = False


def debug(msg):
    if DEBUG and not QUIET:
        click.echo(click.style("[DEBU] " + msg, fg="green"), err=True)


def error(msg):
    if not QUIET:
        click.echo(click.style("[ERRO] " + msg, fg="red"), err=True)


def warn(msg):
    if not QUIET:
        click.echo(click.style("[WARN] " + msg, fg="yellow"), err=True)


def halt(msg):
    error(msg)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)


def require(pred, msg=None):
    if not pred:
        if not QUIET:
            if msg:
                click.echo(click.style("[INTERNAL ERROR] " + msg, fg="red"), err=True)
            else:
                click.echo(click.style("[INTERNAL ERROR]", fg="red"), err=True)
    assert pred

