#include <stdio.h>
#define MaxSize 100
 
/*
* Created by HarvestWu on 2018/07/18.
*/
using namespace std;
typedef int ElemType;
 
//边表
typedef struct ArcNode
{
	int adjvex;                 //该边所指向的结点的位置
	struct ArcNode *nextarc;    //指向下一条边的指针
	int info;                   //
}ArcNode;
 
//顶点表
typedef struct
{
	char data;                  //顶点信息
	ArcNode *firstarc;          //指向第一条边的指针
}VNode;
 
//邻接表
typedef struct
{
	VNode adjlist[MaxSize];
	int n, e;                    //顶点数、边数
}AGraph;                        //图的邻接表类型
 
//图的深度优先搜索遍历(DFS)
//假设用邻接表作为图的存储结构
 
int visit[MaxSize];
void DFS(AGraph *G, int v)
{
	ArcNode *p;
	visit[v] = 1;					//置标志位1代表已访问
	p = G->adjlist[v].firstarc;		//p指向顶点v的第一条边
	while (p != NULL)
	{
		if (visit[p->adjvex] == 0)	//未访问则递归访问
			DFS(G, p->adjvex);
		p = p->nextarc;
	}
 
}
 
//图的广度优先搜索遍历(BFS)
//假设用邻接表作为图的存储结构
 
void BFS(AGraph *G, int v, int visit[MaxSize])
{
	ArcNode *p;
	int que[MaxSize], front = 0, rear = 0;
	int j;
	visit[v] = 1;				//置标志位1代表已访问
	rear = (rear + 1) % MaxSize;
	que[rear] = v;				//当前顶点入队，便于此层扫描完后，继续下一层
	while (front != rear)		//队空则遍历结束
	{
		front = (front + 1) % MaxSize;
		j = que[front];
		p = G->adjlist[j].firstarc;
		while (p != NULL)
		{
			if (visit[p->adjvex] == 0)
			{
				visit[p->adjvex] = 1;
				rear = (rear + 1) % MaxSize;
				que[rear] = p->adjvex;
			}
			p = p->nextarc;
		}
	}
}
 
//创建无向图的邻接表
void createAGraph2(AGraph *&AG)
{
	int i, j, k;
	ArcNode *q;
	cout << "输入顶点数、边数:" << endl;
	cin >> AG->n >> AG->e;
	for (i = 0; i<AG->n; i++)
	{
		AG->adjlist[i].data = i;
		AG->adjlist[i].firstarc = NULL;
	}
	cout << "输入边(vi,vj)的顶点序号i,j:" << endl;
	for (k = 0; k<AG->e; ++k)
	{
		cin >> i >> j;
		//头插法
		q = (ArcNode*)malloc(sizeof(ArcNode));
		q->adjvex = j;
		q->nextarc = AG->adjlist[i].firstarc;
		AG->adjlist[i].firstarc = q;
 
		q = (ArcNode*)malloc(sizeof(ArcNode));
		q->adjvex = i;
		q->nextarc = AG->adjlist[j].firstarc;
		AG->adjlist[j].firstarc = q;
 
	}
}
 
//判断图两顶点间是否有路径
int trave(AGraph *G,int a,int b)
{
    for(int i=0;i<G->n;++i)
        visit[i]=0;
    //DFS(G,a);//两种遍历方式均可
    BFS(G,a,visit);
    if(visit[b]==1)
        return 1;
    else return 0;
}
AGraph *AG;
int main()
{
	AG = (AGraph*)malloc(sizeof(AGraph));
	createAGraph2(AG);
	int a,b;
	cout<<"请输入顶点a,b："<<endl;
	while(cin>>a>>b)
        cout<<trave(AG,a,b)<<endl;
 
	return 0;
}
