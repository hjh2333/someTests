# import datasets
# dir_path = 'C:\\Users\\t-jiahuihe\\Data\\squad'
# temp = datasets.load_dataset(dir_path, split='train')
# temp = temp.map(lambda e:{'labels':len(e["context"])})
# print(temp)
# print(temp[0])
from numpy import arange

arr1 = arange(16).reshape((2,2,4)) 
print(arr1)

arr2 = arr1.transpose((1,0,2))
print(arr2)