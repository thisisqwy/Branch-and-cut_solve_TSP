import gurobipy as gp
from gurobipy import GRB
import numpy as np
import tsplib95
import datetime

# 记录开始时间
start_time = datetime.datetime.now()
# ==========================================
# 1. 读取 TSPLIB 算例数据
# ==========================================
file_path = r"D:\Desktop\tsp问题算例\hk48.tsp"
problem = tsplib95.load(file_path)
print("数据是",problem.edge_weights)
# 获取所有节点（TSPLIB 中节点编号通常为 1, 2, ..., n）
cities = list(problem.get_nodes())
n = problem.dimension

# 构建距离字典 cost[(i, j)]，排除自环 (i == j)
cost = {}
for u in cities:
    for v in cities:
        if u != v:
            # tsplib95 内部会自动根据算例类型（如 EUC_2D 等）计算准确的整数或浮点距离
            cost[u, v] = problem.get_weight(u, v)

# ==========================================
# 2. 子回路提取函数 (纯 Python / tuplelist 实现)
# ==========================================
def subtour(edges):
    """根据传入的已选择边集合，提取其中节点数最少的子回路"""
    unvisited = cities[:]
    cycle = cities[:]  # 初始设定为最长可能的回路

    while unvisited:
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            # 使用 Gurobi tuplelist 的 .select(current, '*') 快速匹配出边邻接点
            neighbors = [j for i, j in edges.select(current, "*") if j in unvisited]

        # 寻找点数最少（包含城市最少）的环路，约束的紧致性更高，能够削掉更大的搜索空间
        if len(thiscycle) < len(cycle):
            cycle = thiscycle

    return cycle


# ==========================================
# 3. Callback 函数：动态注入子回路消除约束 (SEC)
# ==========================================
def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        # 1. 获取当前分支定界树节点找到的整数解，model._vars是一个字典，其键是(i,j)，值是变量。而vals键一样，但是值是真的数字。
        vals = model.cbGetSolution(model._vars)

        # 2. 提取 x_ij > 0.5 的边（即当前解选择的边），转为 gp.tuplelist 类型
        selected = gp.tuplelist(
            (i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5
        )

        # 3. 寻找当前解中的最小子回路,其实就是个由顶点组成的列表
        tour = subtour(selected)

        # 4. 如果最小子回路包含的节点数小于总节点数，说明存在非法子圈，添加懒惰约束
        if len(tour) < len(cities):
            # 约束逻辑：子集 tour 内部所有合法边变量的和 <= |tour| - 1
            model.cbLazy(
                gp.quicksum(model._vars[i, j] for i in tour for j in tour if i != j )<= len(tour) - 1)

# ==========================================
# 4. 构建与求解 Gurobi 模型
# ==========================================
m = gp.Model("TSP_TSPLIB")

# 创建决策变量 x_ij，为 0-1 变量,x此时是一个tupledict
x = m.addVars(cost.keys(), vtype=GRB.BINARY, name="x")

# 设置目标函数：最小化总行程距离
m.setObjective(x.prod(cost), GRB.MINIMIZE)

# 约束 1：每个城市必须恰好离开一次（出度 = 1）
m.addConstrs(
    (gp.quicksum(x[i, j] for j in cities if j != i) == 1 for i in cities),
    name="out_degree",
)

# 约束 2：每个城市必须恰好到达一次（入度 = 1）
m.addConstrs(
    (gp.quicksum(x[i, j] for i in cities if i != j) == 1 for j in cities),
    name="in_degree",
)

# 因为model本身没有一个属性是var，所以要设置一个属性让它等于我们的变量x，方便在回调函数中调用。
m._vars = x# 这里不能使用getvar，因为Gurobi 的 model.getVars()是一个用于获取模型中所有决策变量列表的方法

# 【关键配置】开启延迟约束/懒惰约束功能
m.Params.LazyConstraints = 1

# 启动求解并传入 Callback 破圈函数
m.optimize(subtourelim)
# 记录结束时间
end_time = datetime.datetime.now()
# 输出运行总时间
print(f"运行的总时间为： {(end_time - start_time).total_seconds()} 秒.")
# ==========================================
# 5. 输出求解结果与路线顺序
# ==========================================
if m.status == GRB.OPTIMAL:
    print("\n" + "=" * 40)
    print(f"算例名称: {problem.name}")
    print(f"最优总里程 (Objective Value): {m.ObjVal:.2f}")

    # 提取已选择的边
    selected_edges = gp.tuplelist(
        (i, j) for i, j in cost.keys() if x[i, j].X > 0.5
    )

    # 按访问顺序重建完整巡回路线
    current_node = cities[0]
    route = [current_node]
    while len(route) <= len(cities):
        # 寻找从 current_node 出发的下一站
        next_nodes = selected_edges.select(current_node, "*")
        if next_nodes:
            next_node = next_nodes[0][1]
            route.append(next_node)
            current_node = next_node
        else:
            break

    print("最优路线节点访问顺序:")
    print(" -> ".join(map(str, route)))
    print("=" * 40)