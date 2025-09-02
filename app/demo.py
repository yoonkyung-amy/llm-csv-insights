import argparse
from app.utils import load_csv, profile_schema
from app.core import CSVLlmAssistant

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--q", required=True, help="Natural language question about the CSV")
    args = parser.parse_args()

    df = load_csv(args.csv)
    schema = profile_schema(df)
    print("Schema summary:")
    print(schema)

    bot = CSVLlmAssistant(df)
    print("\nAnswer:")
    print(bot.answer(args.q))

if __name__ == "__main__":
    main()
