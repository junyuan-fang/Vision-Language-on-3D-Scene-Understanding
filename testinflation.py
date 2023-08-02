import torch
# test = torch.range(0, 5)
# test2= test.reshape(2, 3)

# test2[0,1]=9

# print(test)
# print(test2)
# print("--------------------------")
# print(test)
# perm = test.permute(1, 0)
# # the following line produces a RuntimeError
# perm.view(2, 3)
# # but this will work
# perm.contiguous().view(2, 3)


# x = torch.range(0,3*3*3*3-1)

# weight = x.reshape(3,3,3,3)
# squize=weight.clone().view(-1,3,3)#height and width -> one single dimension shape> (3,3,3,3) -> (9,3,3)
# print(squize)
# inflated = squize.repeat(3,1,1)#shape>  (9,3,3)  -> (27,3,3)
# print("-----------------------------------------inflation--------------------------------------")
# inflated[:9, :, :] = weight[0, :, :, :].repeat(3, 1, 1)
# inflated[9:18, :, :] = weight[1, :, :, :].repeat(3, 1, 1)
# inflated[18:, :, :] = weight[2, :, :, :].repeat(3, 1, 1)

# print(inflated)
# print(weight[2, :, :, :])


# ##y-axis
# trans = torch.eye(9).repeat(1, 3).contiguous()#(9.27)
# model = torch.range(0,3*3*3*3-1).reshape(3,3,3,3)
# cout, cin = model.shape[:2]
# squize = model.view(cout, cin, -1) #(3.3.9)
# print(trans)
# print(squize)
# weights = torch.matmul(squize, trans)#.permute(2, 1, 0)
# print(weights)
# weights = torch.matmul(squize, trans).permute(2, 1, 0)
# print(weights)


#x-axis
trans = torch.zeros(9, 27)
trans[0, :3] = 1
trans[3, 3:6] = 1
trans[6, 6:9] = 1
trans[1, 9:12] = 1
trans[4, 12:15] = 1
trans[7, 15:18] = 1
trans[2, 18:21] = 1
trans[5, 21:24] = 1
trans[8, 24:27] = 1
trans = trans.contiguous()
model = torch.range(0,3*3*3*3-1).reshape(3,3,3,3)
cout, cin = model.shape[:2]
model = model.view(cout, cin, -1)
weights = torch.matmul(model, trans).permute(2, 1, 0)
print(weights)