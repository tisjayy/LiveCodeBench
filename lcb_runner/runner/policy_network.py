import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# --- 1. State Embedding Model ---
# Use a pre-trained model to create meaningful embeddings from text.
# 'all-MiniLM-L6-v2' is a good starting point: it's fast and effective.
# We'll load it once and reuse it.
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
# The output dimension of this model is 384. We'll embed 3 text fields.
STATE_EMBEDDING_DIM = 384 * 3  # question + code + metadata

class PolicyNetwork(nn.Module):
    """
    A simple policy network that decides how to alter a prompt.
    It now accepts a device for GPU support.
    """
    def __init__(self, input_dim, action_dim, device):
        super(PolicyNetwork, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
        self.to(device)  # Move the network to the specified device

    def forward(self, state_embedding):
        """
        Takes a state embedding and returns a probability distribution over actions.
        """
        # Ensure input is valid
        if torch.isnan(state_embedding).any() or torch.isinf(state_embedding).any():
            # Return uniform distribution as fallback
            return torch.ones(self.fc2.out_features).to(self.device) / self.fc2.out_features
        
        x = F.relu(self.fc1(state_embedding))
        action_scores = self.fc2(x)
        
        # Check for NaN/Inf in action scores before softmax
        if torch.isnan(action_scores).any() or torch.isinf(action_scores).any():
            return torch.ones(self.fc2.out_features).to(self.device) / self.fc2.out_features
            
        return F.softmax(action_scores, dim=-1)

def get_state_embedding(state: dict, device) -> torch.Tensor:
    """
    Converts a state dictionary into a fixed-size semantic tensor using a sentence transformer.
    """
    question = state.get("question", "")
    code = state.get("code", "")
    # Metadata can be a dict, convert it to a string for embedding
    metadata_str = str(state.get("metadata", ""))
    
    # Ensure non-empty strings for embedding
    question = question if question else "empty"
    code = code if code else "empty"
    metadata_str = metadata_str if metadata_str else "empty"

    # --- 2. Semantic Embedding ---
    # Encode each part of the state. The model returns a numpy array, so we convert to a tensor.
    with torch.no_grad():
        question_emb = torch.from_numpy(EMBEDDING_MODEL.encode(question))
        code_emb = torch.from_numpy(EMBEDDING_MODEL.encode(code))
        metadata_emb = torch.from_numpy(EMBEDDING_MODEL.encode(metadata_str))

    # Concatenate the embeddings to form the final state vector
    embedding = torch.cat([question_emb, code_emb, metadata_emb]).to(device)
    
    # Clamp any extreme values
    embedding = torch.clamp(embedding, min=-10, max=10)
    
    return embedding