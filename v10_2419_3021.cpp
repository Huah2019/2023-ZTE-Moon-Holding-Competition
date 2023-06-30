#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll inf = 0x3f3f3f3f3f3f3f3fll;
mt19937 gen(23333);
template <class T>
inline T randInt(T l, T r)
{
    return uniform_int_distribution<T>{l, r}(gen);
}
template <class T>
inline T randReal(T l, T r)
{
    return uniform_real_distribution<T>{l, r}(gen);
}

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
    int w; // 带宽
    int s; // 进入时间
    int t; // 发送所需时间
    ll score;
};

struct Port
{
    int id;
    int w; // 带宽
    int remainW;
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

// 自制线段树模板 begin
struct node
{
    pair<ll, int> val;
    node()
    {
        val = {-inf, 0};
    }
    void merge(const node &u, const node &v)
    {
        val = max(u.val, v.val);
    }
    void push(int l, int r, node &u, node &v)
    {
    }
};
struct segtree
{
    int n;
    vector<node> t;
    segtree(int _n)
    {
        n = _n;
        t = vector<node>(n << 2 | 1);
    }
    template <typename Info>
    void apply(int l, int r, int k, int x, int y, Info &info, function<void(int, int, node &, Info &)> &f)
    {
        if (l >= x && r <= y)
        {
            f(l, r, t[k], info);
            return;
        }
        t[k].push(l, r, t[k << 1], t[k << 1 | 1]);
        int m = (l + r) >> 1;
        if (x <= m)
            apply(l, m, k << 1, x, y, info, f);
        if (y > m)
            apply(m + 1, r, k << 1 | 1, x, y, info, f);
        t[k].merge(t[k << 1], t[k << 1 | 1]);
    }
    template <typename Info>
    void apply(int l, int r, Info &info, function<void(int, int, node &, Info &)> &f)
    {
        apply(1, n, 1, l, r, info, f);
    }
    template <typename Info>
    void apply(int x, Info &info, function<void(int, int, node &, Info &)> &f)
    {
        apply(1, n, 1, x, x, info, f);
    }
    node query(int l, int r, int k, int x, int y)
    {
        if (l >= x && r <= y)
            return t[k];
        t[k].push(l, r, t[k << 1], t[k << 1 | 1]);
        int m = (l + r) >> 1;
        if (x <= m && y > m)
        {
            node ans;
            ans.merge(query(l, m, k << 1, x, y), query(m + 1, r, k << 1 | 1, x, y));
            return ans;
        }
        if (x <= m)
            return query(l, m, k << 1, x, y);
        return query(m + 1, r, k << 1 | 1, x, y);
    }
    node query(int l, int r)
    {
        return query(1, n, 1, l, r);
    }
    // 0 继续搜索子树 1 终止搜索 2 当前结点终止搜索，但父亲结点的儿子继续搜索
    template <typename Info>
    int binary_search(int l, int r, int k, Info &info, function<int(int, int, node &, Info &)> &f, bool left_first)
    {
        int ret = f(l, r, t[k], info);
        if (ret)
            return ret;
        t[k].push(l, r, t[k << 1], t[k << 1 | 1]);
        int m = (l + r) >> 1;
        if (left_first)
            ret = binary_search(l, m, k << 1, info, f, left_first);
        else
            ret = binary_search(m + 1, r, k << 1 | 1, info, f, left_first);
        if (ret == 1)
            return ret;
        if (left_first)
            return binary_search(m + 1, r, k << 1 | 1, info, f, left_first);
        return binary_search(l, m, k << 1, info, f, left_first);
    }
    template <typename Info>
    int find_first(Info &info, function<int(int, int, node &, Info &)> &f)
    {
        return binary_search(1, n, 1, info, f, true);
    }
    template <typename Info>
    int find_last(Info &info, function<int(int, int, node &, Info &)> &f)
    {
        return binary_search(1, n, 1, info, f, false);
    }
};
// 自制线段树模板 end
struct FlowController
{
    segtree t;
    int maxW;
    vector<priority_queue<pair<ll, Flow *>>> q;
    function<void(int, int, node &, pair<ll, int> &)> apply_func = [&](int l, int r, node &u, pair<ll, int> &info)
    {
        u.val = info;
    };
    FlowController(int _maxW) : maxW(_maxW), t(_maxW)
    {
        q.resize(maxW + 1);
    }
    void addFlow(Flow *flow)
    {
        q[flow->w].push({flow->score, flow});
        pair<ll, int> info = {q[flow->w].top().first, flow->w};
        t.apply(flow->w, info, apply_func);
    }
    Flow *findFlow(int w)
    { // 返回流量<=w且分数最大的流，并从控制器中移除这个流
        if (w <= 0)
            return NULL;
        int bestW = t.query(1, w).val.second;
        assert(bestW >= 0 && bestW <= w);
        if (q[bestW].empty())
            return NULL;
        Flow *ans = q[bestW].top().second;
        q[bestW].pop();
        pair<ll, int> info = {-inf, 0};
        if (!q[bestW].empty())
            info = {q[bestW].top().first, bestW};
        t.apply(bestW, info, apply_func);
        return ans;
    }
};

int makeSolution(int cofA, int cofB)
{
    for (auto &flow : flows)
        flow.score = 1ll * flow.t * cofA + flow.w * cofB;
    sort(flows.begin(), flows.end(), [&](Flow &x, Flow &y)
         { return x.s < y.s; });
    for (auto &port : ports)
        port.remainW = port.w;
    int maxW = 0;
    for (auto &port : ports)
        maxW = max(maxW, port.w);
    FlowController fc(maxW);
    int fID = 0;
    priority_queue<int, vector<int>, greater<int>> timePoint;
    const int B = 1e7;
    bitset<B> vis1;
    set<int> vis2;
    int score = 1;
    auto addTimePoint = [&](int x)
    {
        score = max(score, x);
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
    for (auto &flow : flows)
        addTimePoint(flow.s);
    struct ReleasePoint
    {
        int t, portID, w;
        bool operator<(const ReleasePoint &o) const
        {
            return t > o.t;
        }
    };
    priority_queue<ReleasePoint> flowReleasePoint;
    while (!timePoint.empty())
    {
        int t = timePoint.top();
        timePoint.pop();
        // cout << t << '\n';
        while (!flowReleasePoint.empty() && flowReleasePoint.top().t <= t)
        {
            auto x = flowReleasePoint.top();
            flowReleasePoint.pop();
            ports[x.portID].remainW += x.w;
        }
        while (fID < (int)flows.size() && flows[fID].s <= t)
        {
            fc.addFlow(&flows[fID++]);
        }
        vector<int> pID(ports.size());
        iota(pID.begin(), pID.end(), 0);
        shuffle(pID.begin(), pID.end(), gen);
        for (int portID : pID)
        {
            while (true)
            {
                Flow *f = fc.findFlow(ports[portID].remainW);
                if (f == NULL)
                    break;
                tasks.push_back({f->id, ports[portID].id, t});
                ports[portID].remainW -= f->w;
                flowReleasePoint.push({t + f->t, portID, f->w});
                addTimePoint(t + f->t);
            }
        }
        // while (true)
        // {
        //     int portID = 0;
        //     for (auto &port : ports)
        //         if (port.remainW > ports[portID].remainW)
        //             portID = port.id;
        //     Flow *f = fc.findFlow(ports[portID].remainW);
        //     if (f == NULL)
        //         break;
        //     tasks.push_back({f->id, ports[portID].id, t});
        //     ports[portID].remainW -= f->w;
        //     flowReleasePoint.push({t + f->t, portID, f->w});
        //     addTimePoint(t + f->t);
        // }
    }
    return score;
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
struct scoreCalculator
{
    vector<double> scores;
    void add(int costTime)
    {
#ifdef RUN_IN_LOCAL
        costTime = max(1, costTime);
        double score = 100 / (mylog(costTime) / mylog(10));
        scores.push_back(score);
        cout << "score: " << score << '\n';
#endif
    }
    void showAveScore()
    {
#ifdef RUN_IN_LOCAL
        if (!scores.empty())
        {
            double totScore = 0;
            for (auto &val : scores)
                totScore += val;
            cout << "aveScore: " << totScore / scores.size() << '\n';
        }
#endif
    }
} sc;

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

    int bestScore = makeSolution(1e9, 1);
    bestTasks = tasks;

    for (int i = 1; i <= 10; ++i)
    {
        tasks.clear();
        int A = randInt<int>(1, 1000000);
        int B = randInt<int>(1, 1000);
        int score = makeSolution(A, B);
        if (score < bestScore)
        {
            bestScore = score;
            bestTasks = tasks;
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
    sc.showAveScore();
}
/*
0
score: 24.8878
1
score: 25.9606
2
score: 25.2395
3
score: 21.6066
4
score: 24.9551
5
score: 19.381
6
score: 20.8792
7
score: 30.1116
8
score: 24.3015
9
score: 24.6092
10
aveScore: 24.1932
*/