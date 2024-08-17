import asyncio
from langchain_community.document_loaders import MongodbLoader

async def load_mongodb_data():
    loader = MongodbLoader(
        connection_string="mongodb://localhost:27017/",
        db_name="MidAssist",
        collection_name="doctor_data",
        field_names=["name", "hospital", "special_area", "contact_number"]
    )
    data = await loader.load()
    return dict(data)

# Run the async function using asyncio.run()
if __name__ == "__main__":
    data = asyncio.run(load_mongodb_data())
    print(data)