import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--reindex", help="Reindex all database", action="store_true", default=False
)
parser.add_argument(
    "--dev", help="Enable developing mode", action="store_true", default=False
)
args = parser.parse_args()


# class Arguments:
#     def __init__(self):
#         self.reindex = False
#         self.dev = False


# args = Arguments()
