### Model Acrhitecture

-----

1. **U-Net**

   **Loss Function** 
   $$
   L_{SEG}(G) =- ( y* log{G({x})} + (1-y) * log{(1-G(x))})
   $$
   Loss 함수는 Cross-Entropy의 형태를 가지게 된다.  $L_{seg}(G)$ 를 최소화하는 방향이 Y=1일 때, G(x)도 1이 되고, y=0일 때, G(x)도 0으로 가는 방향이다. 이러한 function은 이론적으로 Convex하기 때문에, 즉 수렴할 수 있기 때문에 이러한 Loss Function을 쓴다. 

   ㅡ

2. **V-GAN**

   

   V-GAN은 U-Net에 다가, 이 형태가 잘된 Segmentation인지 아닌지를 확인하는 GAN Loss를 추가한 형태이다. 
   $$
   L_{GAN}(G,D) = E_{x,y~P_{data}(x,y)}[logD(x,y)] + E_{x~P_{data}(x)}[log(1-D(x,G(x)))] \\
   
   G^* = argmin_{G}[max_{D}L_{GAN}(G,D)] + \lambda L_{SEG}(G)
   $$
   D(판별함수)가 이 세그멘테이션이 완벽하다고 하면 1에 가까워지고, 아니면 0에 가까워지는 Loss이다.

   

3. **Pic2Pic**
   $$
   \frac{}{}x = x + 1
   $$
    

![](/home/ksj/projects/darkflow/preview.png)



| qwe  | qwe  | qwe  |
| ---- | ---- | ---- |
| qwe  | qwe  | qwe  |



1. 
2. 3
3. 33
   1. 12321
   2. 123

````python
def x(a,b):
    print("a")
````

