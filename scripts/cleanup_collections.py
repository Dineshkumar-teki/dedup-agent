import chromadb


def main() -> None:
    client = chromadb.PersistentClient(path="chroma_db")

    try:
        client.delete_collection("test_collection")
        print("Deleted test_collection")
    except Exception as exc:
        print(f"Could not delete test_collection: {exc}")

    try:
        collection = client.get_collection("sql_enriched_questions")
        count = collection.count()
        print(f"sql_enriched_questions document count: {count}")
    except Exception as exc:
        print(f"Could not inspect sql_enriched_questions: {exc}")


if __name__ == "__main__":
    main()
