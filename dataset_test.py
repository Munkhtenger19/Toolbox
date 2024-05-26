from torch_geometric.datasets import CoraFull, Coauthor, Amazon, CitationFull, Planetoid

dataset0 = Planetoid(root='./datasets', name='Cora')
dataset1 = CoraFull(root='./datasets')
dataset2 = Coauthor(name = 'CS', root='./datasets')
dataset3 = Coauthor(name = 'Physics', root='./datasets')
dataset4 = Amazon(name = 'Computers', root='./datasets')
dataset5 = Amazon(name = 'Photo', root='./datasets')
dataset6 = CitationFull(name = 'Cora', root='./datasets')
dataset7 = CitationFull(name = 'DBLP', root='./datasets')
dataset8 = CitationFull(name = 'PubMed', root='./datasets')

data = [dataset0, dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8]

# data = dataset0[0]
print('s',dataset4.y.shape[0])

for d in data:
    print(d[0])