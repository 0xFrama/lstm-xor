import numpy as np
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 3
sequence_length = 50
batch_size = 100
learning_rate = 0.01

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=1, num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size,self.hidden_size,self.num_layers,batch_first=True)
        self.fc = nn.Linear(self.hidden_size,1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size,dtype=torch.float32).to(device)
        c0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size,dtype=torch.float32).to(device)

        ht, _ = self.lstm(x, (h0,c0))
        out = self.fc(ht)
        out = self.sigmoid(out)
        return out


def genDataset(n_samples,sequence_length):
    bstrs = np.random.randint(2,size=(n_samples,sequence_length,1))
    labels = bstrs.cumsum(axis=1) % 2
    return bstrs.astype(np.float32), labels.astype(np.float32)

def randGenDataset(n_samples):
    list_bstrs = [np.random.randint(2,size=(np.random.randint(50)+1,1)).astype(np.float32) for _ in range(n_samples)]
    npy_bstrs = np.array(list_bstrs, dtype=object)
    labels = [npy_bstrs[ind].cumsum(axis=1) % 2 for ind in range(npy_bstrs.shape[0])]
    labels = np.array(labels, dtype=object)
    return npy_bstrs, labels


model = LSTM().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_samples_train = 80000
n_samples_test = 20000
train_bstrs, train_labels = genDataset(n_samples_train,sequence_length) 
test_bstrs, test_labels =  genDataset(n_samples_test,sequence_length)

train_dataset = [(bstr,label) for bstr,label in zip(train_bstrs,train_labels)]
test_dataset = [(bstr,label) for bstr,label in zip(test_bstrs,test_labels)]

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

# Train
n_total_steps = len(train_loader)
model.train()
for epoch in range(num_epochs):
    for n_batch, (bstrs, labels) in enumerate(train_loader):
        n_correct = 0
        n_samples = 0
        bstrs = bstrs.to(device)
        labels = labels.to(device)

        predictions = model(bstrs)
        loss = criterion(predictions,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = torch.where(predictions>0.5,1.,0.)    
        acc = (predictions == labels).type(torch.FloatTensor).mean().item()

        if (n_batch+1) % 800 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{n_batch+1}/{n_total_steps}], Loss: {loss.item():.4f}, Accuracy: {(acc*100):.2f}%')

# Test
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        predicted = model(images)

        predicted = torch.where(predicted>0.5,1.,0.)    
        n_correct += (predicted == labels).type(torch.FloatTensor).mean().item()
    
    print(f'Accuracy of the network on the test set: {(acc*100):.2f}%')






