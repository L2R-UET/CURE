from experiment import run_experiment

def main():
    #Demo for METARBIC dataset
    metrics, model, _ = run_experiment()
    print("Final Results")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()