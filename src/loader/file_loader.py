from langchain_community.document_loaders import DirectoryLoader,CSVLoader
# Load the CSV file
def load_csv_file():
    loader = CSVLoader(
        file_path="../data/train.csv",
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
            "fieldnames": ["qtype", "Question", "Answer"],
        }, 
    )
    data = loader.load()
    print(data)
    return data

