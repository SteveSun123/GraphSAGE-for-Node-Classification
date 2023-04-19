import torch
import torch.nn.functional as F
# 导入GCN层，GraphSAGE层和GAT层
from torch_geometric.nn import GCNConv, SAGEConv, GATConv   
from torch_geometric.datasets import planetoid

# 加载数据
def LoadData():
    dataset_Cora = planetoid.Planetoid(root='./Test_PyTorch/data_Cora/Cora', name='Cora')
    # 查看数据的基本情况
    print(dataset_Cora.data) 
    # dataset_CiteSeer = planetoid.Planetoid(root='./Test_PyTorch/data_CiteSeer/CiteSeer', name='CiteSeer')
    # dataset_PubMed = planetoid.Planetoid(root='./Test_PyTorch/data_PubMed/PubMed', name='PubMed')
    return dataset_Cora

# 构建一个GraphSAGE模型
class GraphSAGEnet(torch.nn.Module):
    def __init__(self, feature, hidden, classes) -> None:
        super(GraphSAGEnet, self).__init__()
        self.sage1 = SAGEConv(feature, hidden)  # 隐含层
        self.sage2 = SAGEConv(hidden, classes)  # 完成分类输出
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.sage1(x, edge_index)
        x = F.relu(x)  # 激活函数使用ReLU(x) = max(0, x)，# 对特征进行非线性转换
        x = F.dropout(x, training= self.training)   # 可以防止过拟合
        x = self.sage2(x, edge_index)
        return F.log_softmax(x, dim= 1)   # 通过log_softmax函数得到概率分布


if __name__ == '__main__':
    dataset = LoadData()
    # print(dataset)
    # 模型部署、数据准备、优化器选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGEnet(dataset.num_node_features, 16, dataset.num_classes).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.01, weight_decay= 5e-4)
    
    model.train()   # 模型训练
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()  # 反向传播，迭代训练参数
        optimizer.step()
    
    # 模型评价和分类结果精度
    model.eval()
    _, pred = model(data).max(dim= 1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    print("GraphSAGE Accuracy: {:.4f}".format(acc))  # GraphSAGE Accuracy on Cora: 0.7860

    
    