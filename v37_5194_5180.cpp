#pragma GCC optimize(2)
#include <bits/stdc++.h>
typedef long long ll;
const ll inf = 0x3f3f3f3f3f3f3f3fll;
using namespace std;
const int SEGTREE_NUM = 2;

mt19937 gen(23333);
template <class T>
inline T randInt(T l, T r)
{
    return std::uniform_int_distribution<T>{l, r}(gen);
}
template <class T>
inline T randReal(T l, T r)
{
    return std::uniform_real_distribution<T>{l, r}(gen);
}
double getTime() { return (double)clock() / CLOCKS_PER_SEC; }

//  由于线上用cmath库就会编译错误，只能手动实现这些数学函数... ...
//  泰勒展开求exp，e^x=1+x+x^2/(2!)+x^3/(3!)+...，在牛顿迭代求log函数的过程中需要用到
double myexp(double x)
{
    double ans = 1;
    double z = 1;
    int i = 1;
    while (abs(z) >= 1e-6)
    {
        z *= x;
        z /= i;
        ans += z;
        ++i;
    }
    return ans;
}

/*
牛顿迭代：
求方程f(x)=0的解
取x[0]作为初始近似解
x[i+1]=x[i]-f(x)/f'(x)
不断迭代直到|x[t]-x[t-1]|<=想要的精度
x[t]即为牛顿迭代得到的近似解
*/
// 牛顿迭代求sqrt函数，x^2=n => f(x)=x^2-n  f'(x)=2x
double mysqrt(double n)
{
    double x = n; // 取初始解为n
    double last;
    do
    {
        last = x;
        x -= (x - n / x) / 2;
    } while (abs(x - last) > 1e-8);
    return x;
}

// 牛顿迭代求以e为底数的log函数，e^x=n => f(x)=e^x-n f'(x)=e^x
double mylog(double n)
{
    double x = 22; // 2.7^22 越3e9左右，一定大于总花费时间（即一定满足n<=3e9），因此取初始解为22
    double last;
    do
    {
        last = x;
        x -= (1 - n / myexp(x));
    } while (abs(last - x) > 1e-6);
    return x;
}

struct Flow
{
    int id;
    int w;  // 带宽
    int s;  // 进入时间
    int t;  // 发送所需时间
    int rk; // 流按w为第一关键字，id为第二关键字的排名
    int type;
    int score[SEGTREE_NUM];
};
vector<int> minRk; // minRk[i],流量>=i的流的最小rk
vector<int> maxRk; // maxRk[i],流量<=i的流的最大rk
int maxW;          // 最大端口流量

struct Port
{
    int id;
    int w; // 带宽
    int remainW;
    // 后面这部分的lastAddTime可以超过当前实际t，用于提前模拟端口队列中的流加入和释放
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> flows;
    int lastAddTime;
    int lastRemainW;
    void addFlow(Flow *f)
    {
        while (lastRemainW < f->w)
        {
            assert(!flows.empty());
            lastAddTime = flows.top().first;
            lastRemainW += flows.top().second;
            flows.pop();
        }
        flows.push({lastAddTime + f->t, f->w});
        lastRemainW -= f->w;
    }
    void popUntil(int t)
    {
        while (!flows.empty() && flows.top().first <= t)
        {
            lastAddTime = flows.top().first;
            lastRemainW += flows.top().second;
            flows.pop();
        }
    }
    void reset()
    {
        lastAddTime = 0;
        lastRemainW = w;
    }
    // 用于提前模拟端口队列中的流加入和释放
};

struct Task
{
    int flowID;
    int portID;
    int sendTime;
};

vector<Flow> flows;
vector<Port> ports;
vector<Task> tasks;
vector<Task> bestTasks;
pair<int, int> bestScore;

void clearState()
{
    flows.clear();
    ports.clear();
    tasks.clear();
}
int to_int(string s)
{
    int ans = 0;
    for (char c : s)
    {
        assert(c >= '0' && c <= '9');
        ans = ans * 10 + c - '0';
    }
    return ans;
}
void getlineSplit(ifstream &in, vector<char> splitChars, vector<string> &words)
{
    words.clear();
    string line;
    getline(in, line);
    string word;
    for (char c : line)
    {
        bool isSplitChar = false;
        for (char splitChar : splitChars)
            if (c == splitChar)
                isSplitChar = true;
        if (isSplitChar)
        {
            if (!word.empty())
                words.push_back(word);
            word = "";
        }
        else
            word += c;
    }
    if (!word.empty())
        words.push_back(word);
}
void readFlows(ifstream &flowIn)
{
    string firstLine;
    vector<string> words;
    getline(flowIn, firstLine);
    while (true)
    {
        getlineSplit(flowIn, {',', ' '}, words);
        if (words.empty())
            break;
        assert(words.size() == 4);
        flows.push_back({to_int(words[0]), to_int(words[1]), to_int(words[2]), to_int(words[3])});
    }
}
void readPorts(ifstream &portIn)
{
    string firstLine;
    vector<string> words;
    getline(portIn, firstLine);
    while (true)
    {
        getlineSplit(portIn, {',', ' '}, words);
        if (words.empty())
            break;
        assert(words.size() == 2);
        ports.push_back({to_int(words[0]), to_int(words[1])});
    }
}

// 线段树结点，自制模板
struct node
{
    pair<int, int> minVal, maxVal;
    node()
    {
        maxVal = {-1e9, -1};
        minVal = {1e9, -1};
    }
    void merge(const node &u, const node &v)
    {
        maxVal = max(u.maxVal, v.maxVal);
        minVal = min(u.minVal, v.minVal);
    }
    void push(int l, int r, node &u, node &v)
    {
    }
    void reset()
    {
        maxVal.first = -1e9;
        maxVal.second = -1;
        minVal.first = 1e9;
        minVal.second = -1;
    }
};
// 写阶段二的时候临时想出来的线段树优化，因为只有单点修改，所以可以改成非递归的形式
struct Segtree
{
    int n, lgn;
    vector<node> t;
    vector<vector<int>> longestSegmentID;
    vector<vector<int>> longestSegmentLen;
    Segtree() { n = 0; }
    void reset(int _n)
    {
        int lastN = n;
        n = _n;
        if (n < 1)
            n = 1;
        if ((n & -n) != n)
            n = 1 << (__lg(n) + 1);
        if (n != lastN) // 不相等则重新分配内存，这样最多只需要做10次预处理
        {
            /*
            原理：
                开一颗有n个叶子结点的满二叉树(n必须是2的幂次才能开满二叉树)，编号为1,2,...,2n-1
                结点i的两个儿子为2*i,2*i+1（如果存在的话），结点i的父亲为i/2
                叶子结点编号为n-1,n,...,2n-1，序列中第x个元素对应的叶子结点编号为n-1+x
                因此单点修改操作很简单，直接获取叶子结点编号进行修改，然后不断更新父亲结点信息即可
                对于区间查询操作，假设要查[l,r]的信息（记为info(l,r)），由于线段树中每个结点代表一个区间信息，假设[l,x]为
                满足x<=r的最长的区间，这个区间对应的结点编号为id，那么区间[l,r]的信息为mergeInfo(info(l,x),info(x+1,r))
                对info(x+1,r)仍然可以递归采用相同的方法，因此每次需要对于一段区间[l,r]，找出一个这样的x，
                满二叉树有一个良好的性质，就是每个结点维护的线段都是一个2的幂次，即x-l+1一定是2的幂次，
                设k为最大的非负整数满足2^k<=r-l+1，那么只需要对每个l记录两个信息longestSegmentID[l][k]和longestSegmentLen[l][k]
                分别表示以l为左端点，长度<=2^k的最长区间的结点编号和最长长度，这部分预处理的实际复杂度为O(nlogn)

            */
            lgn = __lg(n);
            t = vector<node>(2 * n);
            longestSegmentID = vector<vector<int>>(n + 1, vector<int>(lgn + 1));
            longestSegmentLen = vector<vector<int>>(n + 1, vector<int>(lgn + 1));

            int id = 1;
            for (int len = n, lglen = lgn; len >= 1; len >>= 1, --lglen)
            {
                for (int i = 1; i <= n; i += len)
                {
                    longestSegmentLen[i][lglen] = len;
                    longestSegmentID[i][lglen] = id++;
                }
            }
            for (int i = 1; i <= n; ++i)
                for (int j = 1; j <= lgn; ++j)
                    if (!longestSegmentID[i][j])
                    {
                        longestSegmentID[i][j] = longestSegmentID[i][j - 1];
                        longestSegmentLen[i][j] = longestSegmentLen[i][j - 1];
                    }
        }
        else // 相等只需reset所有结点信息
        {
            for (node &val : t)
                val.reset();
        }
    }
    // 修改第x(1<=x<=n)个位置的信息，一次操作的时间复杂度为O(logn)
    template <typename Info>
    void apply(int x, Info &info, function<void(int, int, node &, Info &)> &f)
    {
        int u = n - 1 + x;
        f(x, x, t[u], info);
        u >>= 1;
        while (u)
        {
            t[u].merge(t[u << 1], t[u << 1 | 1]);
            u >>= 1;
        }
    }
    // 查询区间[l,r]的信息，一次操作的时间复杂度为O(logn)
    node query(int l, int r)
    {
        int k = __lg(r - l + 1);
        node ans = t[longestSegmentID[l][k]];
        l += longestSegmentLen[l][k];
        while (l <= r)
        {
            int k = __lg(r - l + 1);
            ans.merge(ans, t[longestSegmentID[l][k]]);
            l += longestSegmentLen[l][k];
        }
        return ans;
    }
};
// 流管理器，管理所有在调度区的流
struct FlowController
{
    int sz[SEGTREE_NUM];    // 每颗线段树中当前有多少条流
    Segtree t[SEGTREE_NUM]; // 线段树，对每种类型的流都开一颗线段树
    int flowNum;            // 最多有多少条流
    vector<Flow *> flows;
    vector<int> st; // 第i条流在哪些线段树里存在，用二进制位状态表示
    // 线段树区间修改函数
    function<void(int, int, node &, pair<pair<int, int>, pair<int, int>> &)> apply_func = [&](int l, int r, node &u, pair<pair<int, int>, pair<int, int>> &info)
    {
        u.minVal = info.first;
        u.maxVal = info.second;
    };
    void reset(int _flowNum)
    {
        flowNum = _flowNum;
        for (int i = 0; i < SEGTREE_NUM; ++i)
        {
            t[i].reset(flowNum);
            sz[i] = 0;
        }
        if ((int)flows.size() != flowNum)
        {
            flows = vector<Flow *>(flowNum + 1, NULL);
            st = vector<int>(flowNum + 1, 0);
        }
        else
        {
            for (Flow *&val : flows)
                val = NULL;
            for (int &val : st)
                val = 0;
        }
    }
    // 插入一条流，tBitset用来表示插入到哪些线段树里
    void addFlow(Flow *flow, int tBitset)
    {
        assert(flow->rk >= 1 && flow->rk <= flowNum);
        for (int i = 0; i < SEGTREE_NUM; ++i)
            if (tBitset >> i & 1)
            {
                pair<int, int> infoMin = {flow->score[i], flow->rk};
                pair<int, int> infoMax = {flow->score[i], flow->rk};
                auto info = make_pair(infoMin, infoMax);
                t[i].apply(flow->rk, info, apply_func);
                ++sz[i];
            }
        flows[flow->rk] = flow;
        st[flow->rk] = tBitset;
    }
    // 删除一条流
    void delFlow(Flow *flow)
    {
        for (int i = 0; i < SEGTREE_NUM; ++i)
            if (st[flow->rk] >> i & 1)
            {
                pair<int, int> infoMin = {1e9, -1};
                pair<int, int> infoMax = {-1e9, -1};
                auto info = make_pair(infoMin, infoMax);
                t[i].apply(flow->rk, info, apply_func);
                --sz[i];
            }
        flows[flow->rk] = NULL;
        st[flow->rk] = 0;
    }
    // 第segtree_id颗线段树中是否存在<=w的流
    bool exist(int w, int segtree_id)
    {
        if (maxRk[w] < 1)
            return false;
        return t[segtree_id].query(1, maxRk[w]).maxVal.second != -1;
    }
    // 返回 第segtree_id颗线段树中流量<=w的一个流，并从控制器中移除这个流，如果op=0，返回分数最小的流，否则返回分数最大的流
    Flow *findFlow(int w, int op, int segtree_id)
    {
        if (maxRk[w] < 1)
            return NULL;
        int rk = op ? t[segtree_id].query(1, maxRk[w]).maxVal.second : t[segtree_id].query(1, maxRk[w]).minVal.second;
        if (rk == -1)
            return NULL;
        Flow *ans = flows[rk];
        delFlow(ans);
        return ans;
    }
} fc;

double totalScore = 0;

const int B = 1e7;
// 时间管理器，管理所有有效时间点
struct TimePointManager
{
    bitset<B> vis1; //<B的用bitset O(1)维护
    set<int> vis2;  //>=B的用set O(logn)维护，这种情况在当前数据集中不会出现
    priority_queue<int, vector<int>, greater<int>> timePoint;
    void reset()
    {
        vis1.reset();
        vis2.clear();
        while (!timePoint.empty())
            timePoint.pop();
    }
    void add(int x)
    {
        if (x < B)
        {
            if (!vis1[x])
            {
                vis1[x] = 1;
                timePoint.push(x);
            }
        }
        else
        {
            if (!vis2.count(x))
            {
                vis2.insert(x);
                timePoint.push(x);
            }
        }
    };
    bool empty() { return timePoint.empty(); }
    int nextTime() { return timePoint.top(); }
    void popTime() { timePoint.pop(); }
} tpm;

pair<int, int>
makeSolution(double rate,
             function<bool(Flow, Flow)> cmpFlow,
             function<bool(int, int)> &cmpPort,
             bool PRE_ALLO, bool RUN_UNTILE_NULL)
{
    for (auto &p : ports)
        p.reset();
    tasks.clear();
    int waitSizeLimit = ports.size() * 20;
    int portQueSizeLimit = 30;
    vector<queue<Flow *>> portQue(ports.size()); // 端口排队队列
    vector<int> sortedPortID(ports.size());
    iota(sortedPortID.begin(), sortedPortID.end(), 0);
    sort(sortedPortID.begin(), sortedPortID.end(), [&](int x, int y)
         { return ports[x].w > ports[y].w; });

    sort(flows.begin(), flows.end(), cmpFlow);
    for (int i = 0; i < flows.size(); ++i) // 排序后，将流的分数改为整数，对整数操作速度更快
        flows[i].score[0] = flows[i].score[1] = flows.size() - i;

    // type=0的流可以放入端口的等待队列，type=1的流不能放入端口的等待队列
    for (auto &flow : flows)
        flow.type = 0;
    // 上一轮的流丢失的总流量为 bestScore.second * 0.5，根据这个流量，调整两种类型的流的数量
    int tLIM = bestScore.second == 1e9 ? 0 : bestScore.second * 0.5 * (0.6 + rate);
    for (int i = 0; i < flows.size(); ++i)
    {
        if (flows[i].t <= tLIM)
        {
            tLIM -= flows[i].t;
            flows[i].type = 1;
        }
        else
            break;
    }
    sort(flows.begin(), flows.end(), [&](Flow &x, Flow &y)
         { return x.s < y.s; });
    for (auto &port : ports)
        port.remainW = port.w;

    fc.reset(flows.size());
    tpm.reset();

    int fID = 0;
    for (auto &flow : flows)
        tpm.add(flow.s);
    // 流的释放结点
    struct ReleasePoint
    {
        int t, portID, w;
        bool operator<(const ReleasePoint &o) const
        {
            return t > o.t;
        }
    };
    priority_queue<ReleasePoint> flowReleasePoint;
    int score = 1, exScore = 0;
    int remainFlowNum = flows.size();
    while (!tpm.empty())
    {
        int t = tpm.nextTime();
        tpm.popTime();
        score = max(score, t); // 最后一个时间点一定是所有流发送完的总时间
        // 释放流
        while (!flowReleasePoint.empty() && flowReleasePoint.top().t <= t)
        {
            auto x = flowReleasePoint.top();
            flowReleasePoint.pop();
            ports[x.portID].remainW += x.w;
        }
        for (auto &p : ports)
            p.popUntil(t);
        // 将t时刻之前产生的流加入流控制器
        while (fID < (int)flows.size() && flows[fID].s <= t)
        {
            fc.addFlow(&flows[fID], flows[fID].type == 0 ? 2 : 1);
            ++fID;
        }
        // 更新端口排队队列
        for (int portID = 0; portID < (int)ports.size(); ++portID)
        {
            while (!portQue[portID].empty() && ports[portID].remainW >= portQue[portID].front()->w)
            {
                auto x = portQue[portID].front();
                portQue[portID].pop();
                // ports[portID].addFlow(x);
                ports[portID].remainW -= x->w;
                flowReleasePoint.push({t + x->t, portID, x->w});
                tpm.add(t + x->t);
            }
        }
        vector<int> pID(ports.size());
        iota(pID.begin(), pID.end(), 0);
        sort(pID.begin(), pID.end(), cmpPort);
        // 对每个队列为空的端口，加入可以立即发送的流
        for (int portID : pID)
            if (portQue[portID].empty())
            {
                while (true)
                {
                    Flow *f = fc.findFlow(ports[portID].remainW, 1, 0);
                    if (f == NULL)
                        f = fc.findFlow(ports[portID].remainW, 1, 1);
                    if (f == NULL)
                        break;
                    tasks.push_back({f->id, ports[portID].id, t});
                    ports[portID].addFlow(f);
                    ports[portID].remainW -= f->w;
                    flowReleasePoint.push({t + f->t, portID, f->w});
                    tpm.add(t + f->t);
                    --remainFlowNum;
                }
            }

        // 填充端口排队队列，避免丢弃太多流
        auto fillQue = [&](int portID, bool allsize, bool RUN_UNTILE_NULL, bool is_max, int segtree_id)
        {
            auto &p = ports[portID];
            while (RUN_UNTILE_NULL || RUN_UNTILE_NULL == false && fc.sz[1] + (allsize ? fc.sz[0] : 0) > waitSizeLimit)
            {
                if (portQue[portID].size() >= (int)portQueSizeLimit)
                    return;
                int maxW = p.w;
                if (PRE_ALLO)
                    maxW = min(maxW, p.lastRemainW);
                Flow *f = fc.findFlow(maxW, is_max, segtree_id);
                if (f == NULL)
                {
                    if (!PRE_ALLO || !fc.exist(p.w, segtree_id))
                        return;
                    int timePoint = p.flows.top().first;
                    p.popUntil(timePoint);
                    continue;
                }
                tasks.push_back({f->id, p.id, t});
                p.addFlow(f);
                portQue[portID].push(f);
                --remainFlowNum;
            }
        };
        if (remainFlowNum > ports.size() * portQueSizeLimit)
        { // 这个时候只需要尽量满足fc.sz[1]<=waitSizeLimit即可
            for (int spID = 0; spID < (int)sortedPortID.size(); ++spID)
                if (fc.sz[1] > waitSizeLimit)
                {
                    // int portID = sortedPortID[spID];
                    int portID = sortedPortID[sortedPortID.size() - 1 - spID];
                    fillQue(portID, false, RUN_UNTILE_NULL, 1, 1);
                }
        }
        else
        { // 这个时候尽量满足fc.sz[0]+fc.sz[1]<=waitSizeLimit，即最后若个条流能塞队列就塞队列
            for (int spID = 0; spID < (int)sortedPortID.size(); ++spID)
                if (fc.sz[0] + fc.sz[1] > waitSizeLimit)
                {
                    // int portID = sortedPortID[spID];
                    int portID = sortedPortID[sortedPortID.size() - 1 - spID];
                    fillQue(portID, true, RUN_UNTILE_NULL, 1, 1);
                }
        }
        // fillQue(sortedPortID[0], true, false, 1, 1);
        // fillQue(sortedPortID[0], true, false, 1, 0);
        // 丢弃流
        while (fc.sz[0] + fc.sz[1] > waitSizeLimit)
        {
            if ((int)portQue[sortedPortID[0]].size() < portQueSizeLimit)
            { // 如果流量最大的端口的队列还没塞满，则无法丢弃流，还需要继续塞
                if (fc.sz[1])
                {
                    Flow *f = fc.findFlow(maxW, 1, 1);
                    tasks.push_back({f->id, ports[sortedPortID[0]].id, t});
                    portQue[sortedPortID[0]].push(f);
                }
                else
                {
                    Flow *f = fc.findFlow(maxW, 1, 0);
                    tasks.push_back({f->id, ports[sortedPortID[0]].id, t});
                    portQue[sortedPortID[0]].push(f);
                }
                continue;
            }
            Flow *f = fc.findFlow(maxW, 1, 0);
            if (f == NULL)
                f = fc.findFlow(maxW, 1, 1);
            --remainFlowNum;
            exScore += f->t * 2;                                    // 丢弃花费
            tasks.push_back({f->id, ports[sortedPortID[0]].id, t}); // 丢弃的任务，放到任意port的等待队列
        }

        if (score + exScore >= bestScore.first + bestScore.second)
            return {score, exScore};
    }
    return {score, exScore};
}

void writeResult(string dir)
{
    fstream out(dir + "/result.txt", std::fstream::out | std::ios_base::trunc);
    // out << "流id,端口id,开始发送时间";
    for (auto &task : bestTasks)
    {
        out << task.flowID << "," << task.portID << "," << task.sendTime << '\n';
    }
    out.close();
}

#define RUN_IN_LOCAL
struct ScoreCalculator
{
    vector<double> scores;
    void add(pair<int, int> useTime)
    {
#ifdef RUN_IN_LOCAL
        double score = 300 / (mylog(useTime.first + useTime.second) / mylog(10));
        cout << "TotalTime: " << useTime.first + useTime.second << " "
             << "Time: " << useTime.first << " "
             << "exTime: " << useTime.second << " "
             << "score: " << score << '\n';
        scores.push_back(score);
#endif
    }
    void showAvgScore()
    {
#ifdef RUN_IN_LOCAL
        if (!scores.empty())
        {
            double totalScore = 0;
            for (double &val : scores)
                totalScore += val;
            cout << "avgScore: " << totalScore / scores.size() << '\n';
        }
#endif
    }
} sc;

void init()
{
    sort(flows.begin(), flows.end(), [&](Flow &x, Flow &y)
         { return x.w == y.w ? x.id < y.id : x.w < y.w; });
    maxW = 0;
    for (auto &port : ports)
        maxW = max(maxW, port.w);
    /*
        每个流的排名对应其在线段树中的结点编号，使得线段树每个叶子结点只被
        一个流占用，这样就方便了添加和删除操作
        按w从小到大排序使得一段连续的流一定在一段连续的区间
        查询w在[l,r]之间的流的信息，线段树中对应的查询区间就是[minRk[l],maxRk[r]]
    */
    minRk = vector<int>(maxW + 1, flows.size() + 1);
    maxRk = vector<int>(maxW + 1, 0);
    for (int i = 0; i < flows.size(); ++i)
    {
        flows[i].rk = i + 1;
        minRk[flows[i].w] = min(minRk[flows[i].w], i + 1);
        maxRk[flows[i].w] = max(maxRk[flows[i].w], i + 1);
    }
    for (int i = 1; i <= maxW; ++i)
        maxRk[i] = max(maxRk[i], maxRk[i - 1]);
    for (int i = maxW - 1; i >= 0; --i)
        minRk[i] = min(minRk[i + 1], minRk[i]);
}

bool work(string dir)
{
    ifstream flowIn(dir + "/flow.txt");
    ifstream portIn(dir + "/port.txt");
    if (!flowIn.good() || !portIn.good())
        return false;
    clearState();
    readFlows(flowIn);
    readPorts(portIn);
    flowIn.close();
    portIn.close();
    init();

    vector<function<bool(int, int)>> cmpPorts;
    cmpPorts.push_back([&](int x, int y)
                       { return ports[x].w > ports[y].w; });
    cmpPorts.push_back([&](int x, int y)
                       { return ports[x].remainW < ports[y].remainW; });

    int it = 0;
    bestScore = {1e9, 1e9};
    bestTasks.clear();
    pair<int, int> score;

    auto updBest = [&]()
    {
        if (score.first + score.second < bestScore.first + bestScore.second)
        {
            bestScore = score;
            bestTasks = tasks;
        }
    };

    vector<function<bool(Flow, Flow)>> cmpFlows;
    cmpFlows.push_back([&](Flow x, Flow y)
                       {
                           if (abs(1.0 * x.w / x.t - 1.0 * y.w / y.t) > 0.1)
                               return 1.0 * x.w / x.t > 1.0 * y.w / y.t;
                           if (x.t != y.t)
                               return x.t < y.t;
                           return x.w < y.w; });

    auto F0 = [&](int t, int w)
    { return 1.0 * t * 1e9 + w; };
    // auto F1 = [&](int t, int w)
    // { return -(long long)(1.0 * w / t * 1000) * 1e9 + t; };
    // auto F2 = [&](int t, int w)
    // { return -(long long)(1.0 * w / t * 10000) * 1e9 - w; };
    // auto F3 = [&](int t, int w)
    // { return -(long long)(1.0 * mysqrt(w) / mysqrt(t)) * 1e9 - w; };
    auto FF1 = [&](int t, int w)
    { return 1.0 * (t / 150) * 1e9 - (w / 10); };
    cmpFlows.push_back([&](Flow x, Flow y)
                       { return F0(x.t, x.w) < F0(y.t, y.w); });

    int Z = 50;
    for (auto &cmpFlow : cmpFlows)
    {
        // for (auto cmpPort : cmpPorts)
        auto cmpPort = cmpPorts[1];
        for (int it = 0; it <= Z; ++it)
        {
            score = makeSolution(
                1.6 * it / Z, cmpFlow,
                cmpPort, true, true);
            updBest();
        }
    }

    auto cmpFlow = [&](Flow x, Flow y)
    {
        return FF1(x.t, x.w) < FF1(y.t, y.w);
    };
    {
        Z = 35;
        auto cmpFlow = [&](Flow x, Flow y)
        {
            return FF1(x.t, x.w) < FF1(y.t, y.w);
        };
        auto cmpPort = cmpPorts[1];
        for (int it = 0; it <= Z; ++it)
        {
            score = makeSolution(
                1.6 * it / Z, cmpFlow,
                cmpPort, true, false);
            updBest();
        }
    }

    sc.add(bestScore);

    writeResult(dir);
    return true;
}
int main()
{
    for (int testId = 0;; ++testId)
    {
        cout << testId << '\n';
        if (!work("../data/" + to_string(testId)))
            break;
    }
    sc.showAvgScore();
    cout << "RUN TIME " << getTime() << "s" << '\n';
}
/*
0
TotalTime: 78199 Time: 6917 exTime: 71282 score: 61.3096
1
TotalTime: 710816 Time: 2864 exTime: 707952 score: 51.2667
2
TotalTime: 604441 Time: 6831 exTime: 597610 score: 51.891
3
TotalTime: 295225 Time: 9625 exTime: 285600 score: 54.8431
4
TotalTime: 729234 Time: 19306 exTime: 709928 score: 51.1695
5
TotalTime: 2735089 Time: 20071 exTime: 2715018 score: 46.6058
6
TotalTime: 172047 Time: 10689 exTime: 161358 score: 57.2995
7
TotalTime: 4501954 Time: 32020 exTime: 4469934 score: 45.0897
8
TotalTime: 1041338 Time: 5142 exTime: 1036196 score: 49.8538
9
TotalTime: 978637 Time: 10979 exTime: 967658 score: 50.0783
10
avgScore: 51.9407
RUN TIME 182.711s
*/