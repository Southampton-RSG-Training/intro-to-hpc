import time
from argparse import ArgumentParser

ap = ArgumentParser(description="Dummy Python script to demonstrate submitting jobs")
ap.add_argument("--input", required=True, help="Path to the input dataset")
ap.add_argument("--output", required=True, help="Path to where to save output")
args = ap.parse_args()

print("This is an example script, not much will be done!")
print("Input data:", args.input)
print("Output path:", args.output)
print("Now counting down for 240 seconds...")

for i in range(240, 0, -1):
    print(f"{i}...")
    time.sleep(1)
