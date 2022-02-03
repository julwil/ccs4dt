import json
import time

import requests


def log(text, same_line=False):
    print(text, flush=True, end='\r' if same_line else '\n')


def benchmark(location_id, input_batch, n, max_iter=10):
    log(f"Running benchmark with size {n}...")
    benchmarks = []

    for i in range(max_iter):
        response = requests.post(f"http://localhost:5000/locations/{location_id}/inputs", json=input_batch[:n])
        log("")
        log(f"ITER: {i}/{max_iter}")

        response_json = response.json()
        input_batch_id = response_json["id"]
        start = time.time_ns()
        end = start
        while True:
            response = requests.get(f"http://localhost:5000/locations/{location_id}/inputs/{input_batch_id}")
            response_json = response.json()
            if response_json["status"] == "failed":
                log("Error in API")
                exit(1)

            if response_json["status"] in ["scheduled", "processing"]:
                log(f"Processing... ({(time.time_ns() - start) / 1000000000} s)", same_line=True)
                time.sleep(0.5)
                continue

            if response_json["status"] == "finished":
                end = time.time_ns()
                break

        benchmarks.append(end - start)

    summary = "      BENCHMARK SUMMARY\n" \
              "===============================\n" \
              "\n" \
              f"N:           {n} (input size)\n" \
              f"ITER:        {max_iter} (number of iterations)\n" \
              f"AVG RUNTIME: {(sum(benchmarks) / len(benchmarks)) / 1000000000} s (runtime in seconds)\n" \
              f"RUNTIMES:    {benchmarks} (ns)"

    log(summary)
    log("")
    log("Done")

    with open('benchmark.txt', 'a') as file:
        file.write(summary)
        file.write("\n\n")

    return benchmarks


log("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
log("+                 API BENCHMARKING                   +")
log("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
log("")
log("Let's do this!")

log("")
log("LOCATION CONFIGURATION")
log("======================")
log("")
log("")

file_path = "location_payload.json"  # input("Provide path to location configuration file: ")
log("Loading location configuration...")
file = open(file_path)
location_configuration = json.load(file)
log("done")
log("")

log("Preview:")
log(location_configuration)

log("")
log("")
log("     INPUT BATCH")
log("======================")
log("")

file_path = "synthetic_measurements.json"  # input("Provide path to input batch file: ")
log("Loading input batch ...")
file = open(file_path)
input_batch = json.load(file)
log("done")
log("")

log("Preview:")
for i in range(min(len(input_batch), 10)):
    log(input_batch[i])

log("")
log("")
log("       SETUP")
log("======================")
log("")

log("Creating location configuration...")
response = requests.post("http://localhost:5000/locations", json=location_configuration)
log(f"Response [{response.status_code}]")
log("")

response_json = response.json()
log(response_json)
location_id = response_json["id"]

log("done")


def benchmark_runner(n_list):
    for n in n_list:
        log("")
        log("")
        log(f"  RUNNING BENCHMARK (N={n})")
        log("===============================")
        log("")

        benchmark(location_id, input_batch, n=n, max_iter=1)
        time.sleep(60 * 5)  # Cool down 5 minutes after each benchmark


benchmark_runner([100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 55000, 60000, 65000, 70000, 75000, 80000])
