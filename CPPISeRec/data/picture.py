import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data_file = "ml-1m.txt"  # 修改为您的数据文件名
with open(data_file, "r") as f:
    lines = f.readlines()

# 将数据转换为字典
data_dict = {}
for line in lines:
    user_id, *item_ids = map(int, line.split())
    data_dict[user_id] = item_ids

# 计算每个用户的交互数量
user_interactions = {user_id: len(items) for user_id, items in data_dict.items()}

# 只选取交互次数少于100的用户
filtered_users = {user_id: count for user_id, count in user_interactions.items() if count < 100}

# 按交互数量排序
sorted_users = sorted(filtered_users.items(), key=lambda x: x[1], reverse=True)

# 获取交互数量并重设索引
counts = [count for _, count in sorted_users]


# 统计用户数量
user_count = len(counts)

# 计算横坐标上限，向上取整千
x_upper_limit = ((user_count - 999) // 1000 + 1) * 1000

# 绘制散点图
x_values = range(len(counts[999:x_upper_limit]))
plt.scatter(x_values, counts[999:x_upper_limit])
plt.xlabel("User Ranking")
plt.ylabel("Number of Interactions")

# 设置标题为红色的数据集名称
plt.title(data_file, color='red')

# 设置横坐标刻度并使数字竖着显示
xticks = range(0, x_upper_limit + 1000, 1000)
plt.xticks(xticks, rotation='vertical')

# 设置纵坐标刻度
yticks = range(0, 110, 10)
plt.yticks(yticks)

plt.xlim(0, x_upper_limit)
plt.ylim(0, 100)
plt.show()


interactions_range = range(0, 16)
users_in_range = sum([1 for count in counts if count in interactions_range])

print(f"Number of users with interactions between 0 and 15 (inclusive): {users_in_range}")

