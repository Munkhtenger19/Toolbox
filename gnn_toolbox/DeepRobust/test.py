from torch_geometric.datasets import Planetoid
from DICE import DICE
dataset = Planetoid(root='/datasets', name='Cora')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
attack = DICE()
dataset