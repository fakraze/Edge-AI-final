from huggingface_hub import login, create_repo, upload_folder, HfApi

# Step 1: 登入
login("** your token here **")

# Step 2: 定義 repo 資訊
repo_id = "fakraze/qlora-wikitext2-llama3-3b"
local_model_dir = "qlora-wikitext2"

# Step 3: 確保 repo 存在，若無則建立
api = HfApi()
api.create_repo(repo_id=repo_id, exist_ok=True)  # 這一行很重要！

# Step 4: 上傳本地資料夾
upload_folder(
    repo_id=repo_id,
    folder_path=local_model_dir,
    path_in_repo=".",
    commit_message="Upload QLoRA adapter trained on WikiText-2",
)
