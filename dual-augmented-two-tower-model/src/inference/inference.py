import torch
import yaml
from src.models.dat_model import DATModel
from src.data.loader import load_processed

# Load config and data
with open("configs/default.yaml") as f:
    cfg = yaml.safe_load(f)

data = load_processed(cfg['data_dir'])
mappings = data['mappings']
num_users = len(mappings['users'])  # Fixed: removed .item()
num_items = len(mappings['items'])  # Fixed: removed .item()

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DATModel(num_users, num_items, emb_dim=cfg['emb_dim'], aug_dim=cfg['aug_dim'],
                tower_hidden=cfg['tower_hidden']).to(device)

checkpoint = torch.load("checkpoints/dat_epoch20.pt", map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Get recommendations for user 0
user_id = 0
with torch.no_grad():
    user_tensor = torch.LongTensor([user_id]).to(device)
    user_emb = model.user_tower(
        torch.cat([model.u_emb(user_tensor), model.a_u(user_tensor)], dim=1)
    )
    user_emb = torch.nn.functional.normalize(user_emb, dim=1)
    
    # Get all item embeddings
    item_embeddings = model.item_embeddings(device=device)
    
    # Compute similarity scores
    scores = torch.matmul(user_emb, item_embeddings.T).squeeze(0)
    
    # Get top 5 recommendations
    top_scores, top_indices = torch.topk(scores, 5)
    
    print(f"Top 5 recommendations for user {user_id}:")
    for i, (item_id, score) in enumerate(zip(top_indices, top_scores)):
        print(f"  {i+1}. Item {item_id.item()}: {score.item():.4f}")