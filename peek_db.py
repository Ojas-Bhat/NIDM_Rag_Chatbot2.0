from pymongo import MongoClient
import json
from bson import json_util

def peek_database():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["disaster_hub"]
    collection = db["documents"]

    # Get 2 sample documents
    samples = list(collection.find().limit(2))

    if not samples:
        print("No documents found in the database.")
        return

    print(f"\nFound {collection.count_documents({})} total chunks.\n")

    for i, doc in enumerate(samples):
        print(f"--- SAMPLE {i+1} ---")
        # Don't print the huge embedding vector to keep it readable
        if "embedding" in doc:
            doc["embedding"] = f"[{len(doc['embedding'])} dimensions]"
        print(json.dumps(doc, indent=2, default=json_util.default))
        print("\n")

if __name__ == "__main__":
    peek_database()