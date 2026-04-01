# Fork 完整流程 + 日常开发标准流程

# 一、Fork 后第一次开发流程
```
1. 去 GitHub 点 Fork 
   把原作者项目复制到你自己账号
2. git clone 你自己的仓库
   下载代码到本地
3. git remote add upstream 原作者地址
   配置同步源（以后能更新官方代码）
5. git remote -v 
   查看当前远程
4. git checkout -b dev 
   创建自己的开发分支（永远不在 main 改代码）
5. 在 dev 分支修改代码
6. git add .
   选中所有修改
7. git status  
   查看要提交哪些文件
8. git commit -m "提交说明"
   本地提交
9. git push origin dev
   推送到你自己 GitHub 的 dev 分支
```

# 二、日常开发流程

## 日常开发（改代码 + 提交）
```
1. 确保在 dev 分支
   git checkout dev

2. 修改代码

3. 提交
   git add .
   git commit -m "我改了xxx"

4. 推送到自己 GitHub
   git push origin dev
```

# 三、你以后同步原作者最新代码（官方更新了）
原项目更新了，你想同步到自己项目：

```
1. 切回 main
   git checkout main

2. 拉取原作者最新代码
   git pull upstream main

3. 推送到你自己的 main
   git push origin main

4. 切回 dev 并合并更新
   git checkout dev
   git merge main
```
# 四、最重要的 3 条铁律（永远记住，永不踩坑）
1. **main 分支永远不动**，只用来同步原作者
2. **所有开发都在 dev 分支**
3. **git push origin dev** 只会推到你自己仓库，绝对不会影响原作者

---
# 流程图

flowchart TD
    %% 顶部：GitHub网页操作
    A[GitHub官网<br>点Fork原项目] --> B[得到你自己的仓库<br>konyyds/xxx]
    
    %% 本地初始化
    B --> C[本地克隆自己仓库<br>git clone 你的仓库地址]
    C --> D[添加上游源<br>git remote add upstream 原作者地址]
    
    %% 建开发分支
    D --> E[建专属开发分支<br>git checkout -b dev]
    
    %% 日常开发循环
    E --> F[在dev改代码/调试]
    F --> G[暂存所有修改<br>git add .]
    G --> H[查看将要提交文件<br>git status]
    H --> I[本地提交<br>git commit -m 备注]
    I --> J[推自己云端dev分支<br>git push origin dev]
    J --> F
    
    %% 同步原作者更新（单独分支走）
    K[要同步官方最新代码] --> L[切干净主分支<br>git checkout main]
    L --> M[拉原作者更新<br>git pull upstream main]
    M --> N[更新自己main<br>git push origin main]
    N --> O[切回dev合并更新<br>git checkout dev & git merge main]
    O --> F
    
    %% 红线铁律标注
    style E fill:#e6f7ff
    style L fill:#fff2e8
    style J fill:#f0f8ff
