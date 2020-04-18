import math
class Point():
	def __init__(self,x = 0,y = 0,tp = 0):
		self.x = x   # x-axis
		self.y = y   # y-axis
		self.tp = tp # type(0 or 1)
	def input(self,flag = 0):
		if flag:
			print('x axis, y axis, type(0 or 1): ',end='');
			x,y,tp = input().split()
			self.x,self.y,self.tp = float(x),float(y),int(tp)
		else:
			print('x axis, y axis: ',end='');
			x,y = input().split()
			self.x,self.y = float(x),float(y)
		return self
	def dis(self,point): # distance
		return math.sqrt((self.x-point.x)**2+(self.y-point.y)**2)

if __name__=='__main__':
	print('point number: ',end='');n = int(input())
	point=[Point().input(1) for _ in range(n)];
	print('neighber: ',end='');
	k = int(input())
	p = Point().input()
	class D():
		def __init__(self,dis,pt):
			self.dis = dis
			self.pt = pt
	l = sorted([D(point[i].dis(p),point[i]) for i in range(n)],key = lambda _:_.dis);
	if l[0].pt.x == p.x and l[0].pt.y == p.y:
		print('p(new point) type is %d'%(l[0].pt.tp));
	else:
		type1 = 0
		for i in range(k):
			if l[i].pt.tp:
				type1+=1
		if type1 > k//2:
			print('p(new point) type is 1')
		else:
			print('p(new point) type is 0')
