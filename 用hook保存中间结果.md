## hook分类

1. register_hook -> 用来记录中间导数（只会保存叶子节点，这样就可以获得中间节点的grad）

   - module_obj.register_hook(lambda grad: grad*2)

2. register_backward_hook

3. register_forward_hook

   ```python
   class Perceptron(nn.Module):
       def __init__(self,in_features, hidden_features, out_features):
           nn.Module.__init__(self)
           self.layer1 = nn.Linear(in_features, hidden_features)
           self.layer2 = nn.Linear(hidden_features, out_features)
       def forward(self,x):
           self.layer1_input = x
           x = self.layer1(x)
           self.layer1_output = x
           x = self.layer2(t.relu(x))
           return x
   datas = t.randn((4,3))
   labels = t.randn((4,1))
   perceptron = Perceptron(3,4,1)
   out = perceptron(datas)
   
   
   def show_forward(model, input, output):
       print(input)
       print("__________________________________________________________________")
       print(output)
       
       
   hook2 = perceptron.layer1.register_forward_hook(show_forward)
   hook2.remove()#通过这种方式删除添加的Hook
   ```

   

4. register_pre_forward_hook

