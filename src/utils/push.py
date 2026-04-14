from huggingface_hub import HfApi

api = HfApi()

# This will upload the contents of your "Model" folder
api.upload_folder(
    folder_path="./Model",
    repo_id="adson-silva/gemma3-4b-gsm8k", # Replace with your username and desired model name
    repo_type="model",
)
print("Upload complete!")
