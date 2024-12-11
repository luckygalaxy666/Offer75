
# 目录

<!-- vim-markdown-toc GFM -->

* [1 剪绳子](#1-剪绳子)
* [2 打印1到最大的n位数](#2-打印1到最大的n位数)
* [3 调整奇数到偶数前面](#3-调整奇数到偶数前面)
* [4 统计数组中出现次数超过一半的数字](#4-统计数组中出现次数超过一半的数字)
* [5 最小的K个数](#5-最小的k个数)
* [6 数据流的中位数](#6-数据流的中位数)
* [7 将数组排成最小的数](#7-将数组排成最小的数)
* [8 输出数组中出现次数超过一半的数字 Java](#8-输出数组中出现次数超过一半的数字-java)
* [9 最小的k个数 Java](#9-最小的k个数-java)
* [10 把数组排成最小的数 Java](#10-把数组排成最小的数-java)
* [11 丑数](#11-丑数)
* [12 数组中的逆序对](#12-数组中的逆序对)
* [13 在排序数组中查找数字](#13-在排序数组中查找数字)
* [14 0~n-1的有序数组中缺失的数字](#14-0n-1的有序数组中缺失的数字)
* [15 和为s的两个数字](#15-和为s的两个数字)

<!-- vim-markdown-toc -->

## [1 剪绳子](https://leetcode.cn/problems/jian-sheng-zi-ii-lcof/description/)


计算长度除3的值`K`和余数b``， 根据余数执行对应操作的快速幂

**时间复杂度 O(logK) 空间复杂度 O(1)**
```c++
class Solution {
public:
typedef long long LL;
    LL qmi(LL a,LL k,int mod)
    {
        long long res = 1;
        while(k)
        {
            if(k&1) res = 1ll*res *a % mod;
            a = 1ll*a*a%mod;
            k>>=1;
        }
        return res;
    }
    int cuttingBamboo(int bamboo_len) {
        LL k = bamboo_len /3;
        LL b= bamboo_len %3;
        if(bamboo_len <=3)
            return bamboo_len-1; 
        int mod = 1e9+7;
        if(b == 1)
            return qmi(3,k-1,mod)*4%mod;
        else if (b == 2)
            return qmi(3,k,mod) *2%mod;
        else return qmi(3,k,mod);
    }
}
```

## [2 打印1到最大的n位数](https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/)

考虑大数打印以及前导0  对每一位的数进行dfs递归加入到`vector<string> ans`中

**时间复杂度 O(9^N) 空间复杂度 O(N)**


```c++
class Solution {
public:
    vector<string> ans;
    string tmp;
    void dfs(int pos,int cnt)
    {
        if(pos == cnt) 
        {
            // cout<<pos<<' '<<tmp<<endl;
            ans.push_back(tmp);
            return;
        }
        int start = pos==0?1:0;
        for(int i = start;i<=9;i++)
        {
            tmp.push_back(i+'0');
            //cout<<tmp<<' '<<pos<<endl;
            dfs(pos+1,cnt);
            tmp.pop_back();
        }
    }
    vector<int> countNumbers(int cnt) {
        for(int i = 1;i<=cnt;i++)
            dfs(0,i);
        vector<int> res_int;
        for(int i = 0;i<ans.size();i++)
            res_int.push_back(stoi(ans[i]));
        return res_int;
    }
};
```

## [3 调整奇数到偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

**排序**

**时间复杂度 O(NlogN) 空间复杂度 O(1)**



```c++
class Solution {
public:
 
    vector<int> trainingPlan(vector<int>& actions) {
        sort(actions.begin(),actions.end(),[](int a,int b){return a%2>b%2;});
        return actions;
    }
};
```
## [4 统计数组中出现次数超过一半的数字](https://leetcode.cn/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/description/)

**摩尔投票法**: 记首个元素为n1，众数为x，遍历并统计票数。利用票数和=0可缩小剩余数组区间。当遍历完成时，最后一轮假设的数字为众数


**时间复杂度 O(N) 空间复杂度 O(1)**


```c++
class Solution {
public:
    int inventoryManagement(vector<int>& stock) {
        int ans ;
        int cnt = 0;
        for(int i =0;i<stock.size();i++)
        {
            if(cnt == 0)
            {
                ans = stock[i];
                cnt++;
            }
            else 
                stock[i]==ans?cnt++:cnt--;
        }
        return ans;
    }
};
```

## [5 最小的K个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

**双指针排序 快排 归并排序**

**时间复杂度 O(NlogN) 空间复杂度 O(1)**


```c++
class Solution {
public:
    void quick_sort(vector<int>& q,int l,int r)
    {
        if(l>=r) return;
        int i =l-1,j=r+1;
        int mid = (l+r)>>1;
        int x = q[mid];
        while(i<j)
        {
            do i++;while(q[i]<x);
            do j--;while(q[j]>x);
            if(i<j) swap(q[i],q[j]); 
        }
        quick_sort(q,l,j);
        quick_sort(q,j+1,r);

    }
    void merge_sort(vector<int>&q,int l,int r)
    {
        if(l>=r) return;
        int mid = (l+r)>>1;
        merge_sort(q,l,mid);
        merge_sort(q,mid+1,r);
        int i = l,j = mid +1;
        vector<int> tmp(r-l+1);
        int k = 0;
        while(i<=mid&&j<=r)
        {
            if(q[i]<q[j]) tmp[k++] = q[i++];
            else          tmp[k++] = q[j++];
        }
        while(i<=mid) tmp[k++] = q[i++];
        while(j<=r) tmp[k++] = q[j++];
        for(int i =0;i<r-l+1;i++)
            q[i+l] = tmp[i];

    }
    vector<int> inventoryManagement(vector<int>& stock, int cnt) {
        vector<int> ans;
        // quick_sort(stock,0,stock.size()-1);
        merge_sort(stock,0,stock.size()-1);
        for(int i = 0;i<cnt;i++) ans.push_back(stock[i]);
        return ans;
    }
};
```

## [6 数据流的中位数](https://leetcode.cn/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof/description/)

**双堆对冲**

小根堆维护数据流中较大的一半,大根堆维护较小的一半, 当两个堆的大小相差超过1时，平衡两个堆的大小


```c++
class MedianFinder {
public:
    /** initialize your data structure here. */
    priority_queue<int,vector<int>,greater<int>> min_heap; //记录较大的一半
    priority_queue<int> max_heap;//记录较小的一半
    int n;
    MedianFinder() {
        n = 0;
    }
    
    void addNum(int num) {
        if(min_heap.size()== 0) min_heap.push(num);
        else if (num>=min_heap.top()) min_heap.push(num);
        else max_heap.push(num);
        //调整两个堆的大小,使得两个堆的大小差不超过1
        if(min_heap.size()>max_heap.size()+1){
            max_heap.push(min_heap.top());
            min_heap.pop();
        }
        if(max_heap.size()>min_heap.size()){
            min_heap.push(max_heap.top());
            max_heap.pop();
        }
    }
    
    double findMedian() {
        int n = min_heap.size()+max_heap.size();
        if(n%2==0) return (min_heap.top()+max_heap.top())/2.0;
        else return min_heap.size()>max_heap.size() ? min_heap.top():max_heap.top();
    }
};
```
**时间复杂度 O(logN)**
* 查找中位数 O(1) ： 获取堆顶元素使用 O(1) 时间；
* 添加数字 O(logN) ： 堆的插入和弹出操作使用 O(logN) 时间。

**空间复杂度 O(N)**
* 其中 N 为数据流中的元素数量，小顶堆 A 和大顶堆 B 最多同时保存 N 个元素。



## [7 将数组排成最小的数](https://leetcode.cn/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)

**排序** 

使用如下规则排序：

* 将所有数转化成字符串
* 如果 x + y > y + x 则x应该在y的后面

该规则具备传递性，因此可以对整个数组排序，这里我使用的是快排，也可以用系统的`sort`配上自制规则


```c++
sort(str_pw.begin(),str_pw.end(),[](string a,string b){ return a + b < b + a;});
```

完整快排程序

```c++
class Solution {
public:

    void quick_sort(vector<string> &q,int l,int r)
    {
        if (l>=r) return;
        int i = l-1,j= r+1;
        string x= q[(l+r)>>1];
        while(i<j)
        {
            do i++; while(q[i] +x < x + q[i]);
            do j--; while(q[j] +x > x + q[j]);
            if(i<j) swap(q[i],q[j]);
        }
        quick_sort(q,l,j);
        quick_sort(q,j+1,r);
    }
    string crackPassword(vector<int>& password) {
        vector<string> str_pw(password.size());
        for(int i = 0;i<password.size();i++)
            str_pw[i] = to_string(password[i]);
        quick_sort(str_pw,0,str_pw.size()-1);
        string ans;
        for(int i = 0;i<str_pw.size();i++) ans+=str_pw[i];
        return ans;
    }
};
```

**时间复杂度 O(NlogN)**
**空间复杂度 O(N)**

## [8 输出数组中出现次数超过一半的数字 Java](https://leetcode.cn/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/description/)

**摩尔投票法**: 记首个元素为n1，众数为x，遍历并统计票数。利用票数和=0可缩小剩余数组区间。当遍历完成时，最后一轮假设的数字为众数


```java
class Solution {
    public int inventoryManagement(int[] stock) {
        int cnt = 0;
        int ans = 0;
        for(int num: stock)
        { 
            if(cnt == 0)
            {
                 ans = num;
                 cnt++;
            }
            else 
                (num == ans)?cnt++:cnt--;
        }
        return ans;
    }
}
```
**时间复杂度 O(N)**
**空间复杂度 O(1)**

## [9 最小的k个数 Java](https://leetcode.cn/problems/zui-xiao-de-kge-shu-lcof/)

   我们用一个大根堆实时维护数组的前 cnt 小值。首先将前 cnt 个数插入大根堆中，随后从第 cnt+1 个数开始遍历，如果当前遍历到的数比大根堆的堆顶的数要小，就把堆顶的数弹出，再插入当前遍历到的数。最后将大根堆里的数存入数组返回即可。


```java
class Solution {
    public int[] inventoryManagement(int[] stock, int cnt) {
        int[] ans = new int[cnt];
        if(cnt == 0) return ans;
        // 大根堆 
        PriorityQueue<Integer> heap = new PriorityQueue<Integer>(new Comparator<Integer>()
        {
                public int compare(Integer num1,Integer num2){ return num2 - num1;}
            });

        for(int i = 0 ;i<cnt;i++)
        {
            heap.offer(stock[i]);
        }
        for(int i = cnt ;i<stock.length;i++)
        {
            int u = heap.peek();
            if(stock[i]<u)
            {
                heap.poll();
                heap.offer(stock[i]);
            }
        }
        
        for(int i = 0;i<cnt;i++)
        {
            ans[i] = heap.poll();
        }
        return ans; 

    }
}
```
**时间复杂度 O(NlogK)**
**空间复杂度 O(K)**

## [10 把数组排成最小的数 Java](https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)
**排序**



```Java
//  快排
class Solution {

    void quickSort(String[] q,int l,int r)
    {
        if(l>=r) return;
        int i = l-1;
        int j = r+1;
        String x = q[l+r >>1];
        String tmp = "";
        while(i<j)
        {
            do{i++;}while((q[i] + x).compareTo(x + q[i])<0);
            do{j--;}while((q[j] + x).compareTo(x + q[j])>0);
            if(i<j) 
            {
                tmp = q[i];
                q[i] = q[j];
                q[j] = tmp;
            }
        }
        quickSort(q,l,j);
        quickSort(q,j+1,r);

    }
    public String crackPassword(int[] password) {
        int len = password.length;
        String[] strPass = new String[len];
        for(int i = 0;i<len;i++)
            strPass[i] = "" + password[i];
        
        quickSort(strPass,0,len-1);
        String ans = "";
        for(int i = 0;i<len;i++)
            ans+=strPass[i];
        return ans;
    }
}

```

**时间复杂度 O(NlogN)**
**空间复杂度 O(N)**

## [11 丑数](https://leetcode.cn/problems/chou-shu-lcof/)

**题目描述**

给你一个整数 n ，请你找出并返回第 n 个 丑数 。

说明：丑数是只包含质因数 2、3 和/或 5 的正整数；1 是丑数。

示例 1： 

* 输入: n = 10
* 输出: 12
* 解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。

**解题思路**
   * **动态规划** ： 
        * 丑数的递推性质： 丑数只包含因子 2, 3, 5，因此有 “丑数 = 某较小丑数 x 某因子” （例如：10 = 5 x 2）。
        * 用三个指针分别指向三个因子，每次取三个指针指向的数乘以对应的因子的最小值，作为新的丑数，然后将该丑数对应的因子的指针向后移动一位。
        * **给每个丑数一次×2,×3,×5的机会，如果是三个指针下的最小值，则将最小值对应的那个丑数的机会用掉**
        * 重复上述操作，直到计算第 n 个丑数。

```Java
class Solution {
    public int nthUglyNumber(int n) {
        int[] nums = new int[n];
        int p2 = 0,p3 = 0,p5 = 0;
        nums[0] =1;
        for (int i = 1; i < n; i++) {
            int d2 = nums[p2] * 2 , d3 = nums[p3] * 3 ,d5 = nums[p5] * 5;

            int minx = Math.min(Math.min(d2,d3),d5);

            if(minx == d2) p2++;
            if(minx == d3) p3++;
            if(minx == d5) p5++;
            nums[i] = minx;
        }
        return nums[n-1];
    }
}
```

**时间复杂度 O(N)**
**空间复杂度 O(N)**

## [12 数组中的逆序对](https://leetcode.cn/problems/shu-zu-zhong-de-ni-xu-dui-lcof/description/)

**题目描述**

在股票交易中，如果前一天的股价高于后一天的股价，则可以认为存在一个「交易逆序对」。请设计一个程序，输入一段时间内的股票交易记录 record，返回其中存在的「交易逆序对」总数。

示例 1:

* 输入：record = [9, 7, 5, 4, 6]
* 输出：8
* 解释：交易中的逆序对为 (9, 7), (9, 5), (9, 4), (9, 6), (7, 5), (7, 4), (7, 6), (5, 4)。
 
限制： 0 <= record.length <= 50000


**解题思路**
* **归并排序** ： 
    * 逆序对的数量即为归并排序中的交换次数
    * 在归并排序的过程中，对于两个有序数组，**如果左边数组的元素大于右边数组的元素，则说明左边数组的元素都大于右边数组的元素，逆序对的数量即为左边数组剩余元素的数量**
    * 在归并排序的过程中，每次合并两个有序数组时，计算逆序对的数量，然后合并两个有序数组


```Java
class Solution {
    int cnt = 0;
    public void merge_sort(int[] q,int l,int r)
    {
        if(l>=r) return;
        int mid = l + r >>1;
        merge_sort(q,l,mid) ;merge_sort(q,mid+1,r);
        int i = l, j = mid + 1,k = 0;
        int[] tmp = new int[r-l+1];
        while(i<=mid&& j<=r)
        {
            if(q[i]<=q[j]) tmp[k++] = q[i++];
            else
            {
                tmp[k++] = q[j++];
                cnt +=  mid - i + 1; 
            }
        }
        while(i<=mid) tmp[k++] = q[i++];
        while(j<=r) tmp[k++] = q[j++];
        for(i = l,j = 0;i<=r;i++,j++) q[i] = tmp[j];
    }
    public int reversePairs(int[] record) {
        int l =0 ,r = record.length-1;
        merge_sort(record,l,r-1);
        return cnt;
        
    }
}
```

**时间复杂度 O(NlogN)**
**空间复杂度 O(N)**

## [13 在排序数组中查找数字](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

**题目描述**

统计一个数字在排序数组中出现的次数。

示例 1:

* 输入: nums = [5,7,7,8,8,10], target = 8
* 输出: 2

**解题思路**

* **二分查找** ： 
    * 二分查找找到目标值的左右边界，然后计算出现的次数
    * **找左边界**： mid = l + r >> 1  下取整
    * **找右边界**： mid = l + r + 1 >> 1  上取整

```Java
class Solution {
    public int countTarget(int[] scores, int target) {
        if (scores.length == 0)  return 0;
        int l = 0,r = scores.length -1 ;
        // 找左端点
        while(l<r)
        {
            int mid = l + r >>1; // 下取整 
            if(scores[mid]>=target) r = mid ;
            else l = mid +1;
        }
        int leftNode = r;
        // 找不到
        if (scores[leftNode] != target) return 0;

        // 找右端点
        l = 0 ;r = scores.length - 1;
        while(l<r)
        {
             int mid = l + r + 1 >>1;
             if(scores[mid] <=target) l = mid;
             else r = mid - 1; 
        }
        int rightNode = r;
        
        return rightNode - leftNode + 1;
    }
}
```

**时间复杂度 O(logN)**
**空间复杂度 O(1)**

## [14 0~n-1的有序数组中缺失的数字](https://leetcode.cn/problems/que-shi-de-shu-zi-lcof/)

**题目描述**

一个长度为 n-1 的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围 0～n-1 之内。在范围 0～n-1 内的 n 个数字中有且只有一个数字不在该数组中，请找出这个数字。

示例 1:

* 输入: [0,1,3]
* 输出: 2

**解题思路**

* **二分查找** ： 
    * 二分查找找到第一个不满足条件的数，即nums[mid] != mid
    * 如果nums[mid] == mid 说明缺失的数字在右边，否则在左边



```Java
class Solution {
    public int takeAttendance(int[] records) {
        if(records.length == 0) return 0;
        int last = records.length -1;
        if(records[last] == last) return records.length;
        int l = 0, r = records.length - 1 ;
        while(l<r)
        {
            int mid = l + r >> 1;
            if(records[mid] > mid) r = mid;
            else  l = mid + 1;
        }
        return l;
    }
}
```

**时间复杂度 O(logN)**
**空间复杂度 O(1)**

## [15 和为s的两个数字](https://leetcode.cn/problems/he-wei-sde-liang-ge-shu-zi-lcof/)

**题目描述**

输入一个递增排序的数组和一个数字 s，在数组中查找两个数，使得它们的和正好是 s。如果有多对数字的和等于 s，则输出任意一对即可。

示例 1:

* 输入: nums = [2,7,11,15], target = 9

* 输出: [2,7] 或 [7,2]

**解题思路**

* **双指针** ： 
    * 两个指针分别指向数组的头和尾，如果两个指针指向的数的和大于target，则右指针左移，否则左指针右移



```Java
class Solution {
    public int[] twoSum(int[] price, int target) {
        int l = 0 , r = price.length - 1;
        while(l<r)
        {
            int value = price[l] + price[r];
        
            if(value >target) r--;
            else if(value < target) l++;
            else break;
        }
        int[] ans = {price[l],price[r]};
        return ans;
        
    }
}
```

**时间复杂度 O(N)** 
**空间复杂度 O(1)**
