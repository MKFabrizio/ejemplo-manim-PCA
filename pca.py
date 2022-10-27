from os import fdopen
from tracemalloc import start
from turtle import distance, dot
from manim import *
import numpy as np
import pandas as pd
import math
import random
from numpy import linalg as LA


random.seed(42)

n = 1000
theta = 0.5
mu = [0, 0]
Sigma = [[1, theta], [theta, 1]]
x1, x2 = np.random.default_rng().multivariate_normal(mu, Sigma, n).T
X = list(pd.DataFrame({'x1':x1,'x2':x2}).itertuples())

n = 100
theta = 0.8
mu = [0, 0]
Sigma1 = [[1, theta], [theta, 1]]
x1, x2 = np.random.default_rng().multivariate_normal(mu, Sigma, n).T
x3 = np.random.normal(1.7,0.01,n)
X2 = list(pd.DataFrame({'x1':x1,'x2':x2, 'x3':x3}).itertuples())



class intro(Scene):
    def construct(self):
        title = Text("Análisis de Componentes Principales")
        title.set_color(YELLOW)

        titulo = Title("Análisis de Componentes Principales")

        m1 = Matrix([["x_{11}","x_{12} ","x_{13}","x_{14}","\dots","x_{1p} "],
        ["x_{21}","x_{22} ","x_{23}","x_{24}","\dots","x_{2p} "],
        ["x_{31}","x_{32} ","x_{33}","x_{34}","\dots","x_{3p} "],
        ["x_{41}","x_{42} ","x_{43}","x_{44}","\dots","x_{4p} "],
        ["\\vdots","\\vdots","\\vdots","\\vdots","\\ddots","\\vdots"],
        ["x_{n1}","x_{n2} ","x_{n3}","x_{n4}","\dots","x_{np} "]])

        m2 = Matrix([["z_{11}","z_{12}"],
        ["z_{21}","z_{22}"],
        ["z_{31}","z_{32}"],
        ["\\vdots","\\vdots"],
        ["z_{n1}","z_{n2}"]])


        self.play(FadeIn(title))
        self.wait(1)
        self.play(FadeOut(title))
        self.add(titulo)
        self.wait(3)
        #self.add(m1,m2)
        self.play(FadeIn(m1))
        self.wait(9)
        self.play(ReplacementTransform(m1,m2))

        #m2.generate_target()
        #m2.target.shift(3*LEFT)
        #self.play(MoveToTarget(m2))
        self.wait(5)

class ejemplo1(Scene):
    def construct(self):

        titulo = Title("Veamos un ejemplo")
        #t2 = MathTable(
        #    [["Obs", "edad", "estudios", "altura"],
        #    [1, 35, 5, 171.1],
        #    [2, 41, 7, 170.8],
        #    [ "\\vdots" , "\\vdots", "\\vdots", "\\vdots"],
        #    ["n", 25, 4, 171.2]],
        #    include_outer_lines=True)
        #t22 = MathTable([["Edad"],[35],[28],["\\vdots"],[45]],include_outer_lines=True).scale(0.8).move_to(LEFT*1.5)
        #t21 = MathTable([["Obs"],[1],[2],["\\vdots"],["n"]],include_outer_lines=True).scale(0.8).next_to(t22,LEFT*0.1)
        t22 = MathTable([["Obs","Edad"],[1,35],[2,28],["\\vdots","\\vdots"],["n",45]],include_outer_lines=True).scale(0.8).move_to(LEFT*2.5)
        t23 = MathTable([["Estudios"],[4],[2],["\\vdots"],[7]],include_outer_lines=True).scale(0.8).next_to(t22,RIGHT*0.1)
        t24 = MathTable([["Altura"],[171.1],[170.8],["\\vdots"],[171]],include_outer_lines=True).scale(0.8).next_to(t23,RIGHT*0.1)
        
        
        
        t27 = MathTable([["7.8\\%"]]).scale(0.8).next_to(t23,DOWN*1.5)
        t28 = MathTable([["0.6\\%"]]).scale(0.8).next_to(t24,DOWN*1.5)
        t26 = MathTable([["3.6\\%"]]).scale(0.8).next_to(t27,LEFT*6)
        t25 = MathTable([["CV"]]).scale(0.8).next_to(t26,LEFT*4)
        
        tabla2 = VGroup(t25,t26,t27,t28)
        
        #t2.scale(0.9).move_to(DOWN*0.5)
        #self.add(titulo,t22,t23,t24,tabla2)
 
        self.add(titulo)
        self.wait(3)
        self.play(FadeIn(t22))
        self.play(FadeIn(t23))
        self.play(FadeIn(t24))
        self.wait(2)
        self.play(FadeIn(tabla2))
        self.wait(3)




#EJEMPLO GEOMETRICO
class graf3d(ThreeDScene):
    def construct(self):
    
        axes = ThreeDAxes(
            x_range=(-4,4),
            y_range=(-4,4),
            z_range=(-4,4)
        ).scale(0.8)
        labx = axes.get_x_axis_label(Tex("X"))
        laby = axes.get_y_axis_label(Tex("Y"))
        labz = axes.get_z_axis_label(Tex("Z"))
        dots = [Dot3D(point=axes.coords_to_point(p.x1,p.x2,p.x3), radius=0.05, color=YELLOW) for p in X2]
        self.add(axes, *dots,labx,laby,labz)
        #phi = y con z (hacia arriba)

        plane = Surface(
            lambda u, v: axes.c2p( u, v, 1.7),u_range=[-4,4],v_range=[-4,4],resolution=1)
        plane.set_style(fill_opacity=0.5)
        self.wait(8)
        self.move_camera(phi=75*DEGREES,theta=-45*DEGREES)
        self.wait(2)
        self.play(FadeIn(plane))
        self.wait(3)
        self.move_camera(phi=90*DEGREES, distance=8)
        self.wait(3)
        self.begin_ambient_camera_rotation(rate=0.5)

        self.wait(3)

        self.stop_ambient_camera_rotation()

class generalidades(Scene):
    def construct(self):
        Matriz2 = Matrix([[0.8,0.6],[0.4,0.51],[0.2,0.74],["\\vdots","\\vdots"],[0.4,0.66],["\\vdots","\\vdots"]]).move_to(LEFT*4)
        Matriz = Matrix([[0.8,0.6],[0.4,0.51],[0.2,0.74],["\\vdots","\\vdots"],[0.4,0.66],["\\vdots","\\vdots"]])
        Matriz.add(SurroundingRectangle(Matriz.get_rows()[4]))
        Matriz.move_to(LEFT*4)
        text1=MathTex("x_j").move_to(LEFT*6+DOWN*1.2)
        text2=MathTex("X").move_to(LEFT*4.5+UP*2.9)
        text3=MathTex("Y").move_to(LEFT*3.2+UP*2.9)

        plano = Axes(x_range=(0,1,1),y_range=(0,1,1),y_length=6,x_length=6).move_to(RIGHT*2)
        labx = plano.get_x_axis_label(Tex("X"),direction=DOWN*2+RIGHT)
        laby = plano.get_y_axis_label(Tex("Y"),direction=LEFT*2) 
        vecv = Arrow(plano.get_origin(),plano.c2p(1,1),
        buff=0,color=YELLOW)
        dotx = Dot(point=plano.c2p(0.4,0.66), radius= 0.1, color= BLUE)
        vecx = Arrow(plano.get_origin(),plano.c2p(0.4,0.66),
        buff=0,color=BLUE)
        vecx1 = Arrow(plano.get_origin(),plano.c2p(0.4,0.66),
        buff=0,color=BLUE)

        #compute projection X on V
        numerator = np.dot([0.4,0.66,0], [1,1,0])
        denominator = np.linalg.norm([1,1,0])**2
        scalar = numerator/denominator
        vecProj = scalar*np.array([1,1,0])
        ArrowProj = Arrow(plano.get_origin(),plano.c2p(vecProj[0],vecProj[1]),buff=0,color=PINK)

        #Compute orthogonal line to V
        line = DashedLine(plano.c2p(0.4,0.66), plano.c2p(vecProj[0],vecProj[1]),buff=0, color=WHITE)
        line2 = Line(plano.get_origin(),plano.c2p(1,1))
        Rangle = RightAngle(line2, line, color=GREY)

        label_dot= MathTex("x_j").next_to(dotx,UP*0.1).set_color(BLUE)
        #label_h = MathTex("h_j").next_to(line,RIGHT*0.05+UP*0.01)
        label_vecv = MathTex("v").next_to(vecv,RIGHT*0.05+UP).set_color(YELLOW)
        label_s = MathTex("p").move_to(plano.c2p(vecProj[0]+0.05,vecProj[1])).set_color(PINK)
        label_vecx = MathTex(" \\left \\| x_j \\right \\| ").move_to(plano.c2p(0.2,0.5)).set_color(BLUE)


        #self.add(plano,text1,text2,text3, Matriz,Matriz2,vecx,dotx,vecv,line,Rangle,ArrowProj,label_s,label_dot,label_h,label_vecv,label_vecx)

        self.play(FadeIn(Matriz2),FadeIn(text2),FadeIn(text3))
        self.wait(4)
        self.play(FadeIn(plano,vecv,label_vecv,labx,laby))
        self.wait(3)
        self.play(FadeIn(Matriz),FadeIn(text1),FadeIn(dotx,label_dot))
        self.wait(4)
        self.play(GrowArrow(vecx))
        self.add(label_vecx,vecx1)
        self.wait(5)
        self.play(FadeIn(line))
        self.play(ReplacementTransform(vecx1,ArrowProj))
        self.add(label_s)
        self.wait(5)

class optimizacion(Scene):
    def construct(self):

        plane = Axes(x_range=(0,1,1),y_range=(0,1,1),
        y_length=6,x_length=6
        ).move_to(DOWN*0.5).scale(0.8)
        labx = plane.get_x_axis_label(Tex("X"),direction=DOWN*2+RIGHT)
        laby = plane.get_y_axis_label(Tex("Y"),direction=LEFT*2)        

        x1=[0.3,0.15]
        x2=[0.3,0.5]
        x3=[0.67,0.56]
        dotx1 = Dot(point=plane.c2p(x1[0],x1[1]), radius= 0.1, color= BLUE)
        dotx2 = Dot(point=plane.c2p(x2[0],x2[1]), radius= 0.1, color= PINK)
        dotx3 = Dot(point=plane.c2p(x3[0],x3[1]), radius= 0.1, color= GREEN)

        c = ValueTracker(1)

        vecx1 = Line(plane.get_origin(),plane.c2p(x1[0],x1[1]),buff=0,color=BLUE)
        vecx2 = Line(plane.get_origin(),plane.c2p(x2[0],x2[1]),buff=0,color=PINK)
        vecx3 = Line(plane.get_origin(),plane.c2p(x3[0],x3[1]),buff=0,color=GREEN)
        #Compute projection
         
        def proyector(vecx,c):
            v = [c,math.sqrt(1-math.pow(c,2))]
            vecProjUtoV = (np.dot(vecx, v) / np.linalg.norm(v)**2 ) * np.array(v)  
            return vecProjUtoV[0],vecProjUtoV[1]

        graph2 = always_redraw(lambda: Arrow(plane.coords_to_point(0,0),
        plane.coords_to_point(c.get_value(), math.sqrt(1-math.pow(c.get_value(),2)) ), 
        buff=0,color=YELLOW
        ))

        graph3 = Arrow(plane.c2p(0,0.707),plane.c2p(0.707,0),buff=0,color=BLUE)


        ArrowProj = always_redraw(lambda: Arrow(plane.get_origin(),
        plane.c2p(proyector(x1,c.get_value())[0],proyector(x1,c.get_value())[1]),
        buff=0,color=BLUE
        ))
        h1 = always_redraw(lambda: DashedLine(plane.c2p(x1[0],x1[1]), 
        plane.c2p(proyector(x1,c.get_value())[0],proyector(x1,c.get_value())[1]),
        buff=0, color=BLUE)
        )

        ArrowProj2 = always_redraw(lambda: Arrow(plane.get_origin(),
        plane.c2p(proyector(x2,c.get_value())[0],proyector(x2,c.get_value())[1]),
        buff=0,color=PINK
        ))
        h2 = always_redraw(lambda: DashedLine(plane.c2p(x2[0],x2[1]), 
        plane.c2p(proyector(x2,c.get_value())[0],proyector(x2,c.get_value())[1]),
        buff=0, color=PINK)
        )

        ArrowProj3 = always_redraw(lambda: Arrow(plane.get_origin(),
        plane.c2p(proyector(x3,c.get_value())[0],proyector(x3,c.get_value())[1]),
        buff=0,color=GREEN
        ))
        h3 = always_redraw(lambda: DashedLine(plane.c2p(x3[0],x3[1]), 
        plane.c2p(proyector(x3,c.get_value())[0],proyector(x3,c.get_value())[1]),
        buff=0, color=GREEN)
        )

        labelx1 = MathTex("x_1").next_to(dotx1,RIGHT*0.05+UP).set_color(BLUE)
        labelx2 = MathTex("x_2").next_to(dotx2,RIGHT*0.05+UP).set_color(PINK)
        labelx3 = MathTex("x_3").next_to(dotx3,RIGHT*0.05+UP).set_color(GREEN)


        label_vecv = always_redraw(lambda: MathTex("v").next_to(graph2,RIGHT*0.05+UP).set_color(YELLOW))
        label_vecv2 = MathTex("v_2").next_to(RIGHT*1+DOWN*2.2).set_color(BLUE)

        self.play(
            LaggedStart(DrawBorderThenFill(plane),FadeIn(labx,laby),
            Create(graph2),run_time=5,
            lag_ratio=1)
        )
        self.play(FadeIn(dotx1,dotx2,dotx3,labelx1,labelx2,labelx3,label_vecv))
        self.play(FadeIn(vecx1,vecx2,vecx3))
        self.play(FadeIn(h1,h2,h3))
        self.play(FadeIn(ArrowProj,ArrowProj2,ArrowProj3))
        
        self.play(c.animate.set_value(0),run_time=5,rate_functions=linear)
        self.play(c.animate.set_value(0.85),run_time=4,rate_functions=linear)
        self.play(c.animate.set_value(0.55),run_time=3,rate_functions=linear)
        self.play(c.animate.set_value(0.80),run_time=3,rate_functions=linear)
        self.play(c.animate.set_value(0.707),run_time=2,rate_functions=linear)
        self.play(FadeOut(ArrowProj),FadeOut(ArrowProj2),FadeOut(ArrowProj3),FadeOut(h1),FadeOut(h2),FadeOut(h3),FadeOut(vecx1),FadeOut(vecx2),FadeOut(vecx3),FadeOut(labelx1,labelx2,labelx3))
        label_vecv_v2 = always_redraw(lambda: MathTex("v_1").next_to(graph2,RIGHT*0.05+UP).set_color(YELLOW))
        self.play(ReplacementTransform(label_vecv,label_vecv_v2))
        self.wait(2)
        self.play(GrowArrow(graph3))
        self.add(label_vecv2)
        self.wait(3)

class variables(MovingCameraScene):
    def construct(self):
        tabla1 = MathTable(
            [["Obs",'var 1','var 2','var 3','var 4','var 5','var 6','var 7','var 8','var 9','var 10','var 11','var 12','var 13','var 14','var 15','var 16','var 17','var 18','var 19','var 20','var 21','var 22','var 23','var 24','var 25','var 26','var 27','var 28','var 29','var 30'],
            [1, -1.1 ,  0.35,  0.48, -2.25, -1.03, -0.1 ,  0.69,  0.05,  1.46, 0.97,  1.95,  0.26, -0.53,  0.35,  0.31, -0.46,  1.62, -0.13,-1.51, -0.79,  1.57, -0.64, -0.38, -0.78,  0.66, -0.9 , -0.54,0.77, -0.07,  0.9 ],
            [2, -0.37, -0.13,  0.48, -0.73,  0.17, -0.31, -1.12, -1.82, -0.83,-0.47, -0.09, -2.05,  0.87,  1.73, -0.03, -0.34,  0.63, -0.49,0.25, -1.59, -0.06,  0.56, -1.94,  0.06,  0.57,  0.74,  0.72,1.17, -0.9 ,  0.71],
            [ "\\vdots" , "\\vdots" , "\\vdots" , "\\vdots" , "\\vdots" , "\\vdots" , "\\vdots" , "\\vdots" , "\\vdots" , "\\vdots" , "\\vdots" , "\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\vdots"],
            ["n",1.07,  0.04, -0.69,  0.49, -1.24, -0.42,  0.57, -1.35,  0.66,-0.2 ,  0.12, -0.83, -0.88,  1.66,  0.49,  0.1 , -1.81, -0.35,1.79,  1.16, -0.58,  1.36,  0.19,  1.74, -0.16,  0.29,  0.54,-1.57, -0.32,  0.14]],include_outer_lines=True)
        
        tabla2 = MathTable(
            [["Obs","Componente \\ 1", "Componente \\ 2", "Componente \\ 3", "Componente \\ 4"],
            [1, 0.87, 0.23,-0.98,0.5],
            [2, 0.65 , 0.42,0.01,-0.2],
            ["\\vdots" , "\\vdots", "\\vdots", "\\vdots", "\\vdots"],
            ["n", 0.91, 0.35,-0.25,0.85]],
            include_outer_lines=True)
        tabla2.scale(0.6)
        self.camera.frame.save_state()
        self.camera.frame.move_to(LEFT*38)
        dot_2 = Dot(point=RIGHT*40,radius=0.001).set_color(BLACK)
        dot_1 = Dot(point=RIGHT*1,radius=0.001).set_color(BLACK)

        self.add(tabla1, dot_1, dot_2)
        self.play(self.camera.frame.animate.move_to(dot_2),run_time=7)
        self.play(self.camera.frame.animate.scale(8).move_to(dot_1),run_time=4)
        self.play(Restore(self.camera.frame),run_time=3)
        #self.play(Restore(self.camera.frame))
        self.play(FadeOut(dot_1),FadeOut(dot_2))
        self.wait(3)
        self.play(ReplacementTransform(tabla1,tabla2))
        self.wait(2)



class ejemplo2(Scene):
    def construct(self):
        text1 = Tex("Regresando al ejemplo inicial:")
        text1.to_edge(UP)
        tabla1 = MathTable(
            [["Obs", "edad", "estudios", "altura"],
            [1, 35, 5, 171.1],
            [2, 41, 7, 170.8],
            [ "\\vdots" , "\\vdots", "\\vdots", "\\vdots"],
            ["n", 25, 4, 171.2]],
            include_outer_lines=True)
        tabla1.scale(0.5).move_to(LEFT*3.5)
        tabla2 = MathTable(
            [["Obs","Componente \\ 1", "Componente \\ 2"],
            [1, 0.87, 0.23],
            [2, 0.65 , 0.42],
            ["\\vdots" , "\\vdots", "\\vdots"],
            ["n", 0.91, 0.35]],
            include_outer_lines=True)
        tabla2.scale(0.5).move_to(RIGHT*3.5)
        flecha = Arrow(start=LEFT,end=RIGHT, color=YELLOW,)
        self.add(tabla1,tabla2,flecha,text1)

class general(Scene):
    def construct(self):
        titulo = Title("El caso general con p variables")
        
        matv = Matrix([["x_{12}"],["x_{22}"],["\\vdots"],["x_{n2}"]]).move_to(LEFT)
        matx = Matrix([["x_{11}"],["x_{21}"],["\\vdots"],["x_{n1}"]]).next_to(matv,LEFT)
        matw = Matrix([["x_{13}"],["x_{23}"],["\\vdots"],["x_{n3}"]]).next_to(matv,RIGHT)
        matp = MathTex("\dots").next_to(matw,RIGHT)
        matz = Matrix([["x_{1p}"],["x_{2p}"],["\\vdots"],["x_{np}"]]).next_to(matp,RIGHT)

        matx1 = Matrix([["x_{11}"],["x_{21}"],["\\vdots"],["x_{n1}"]]).next_to(matv,LEFT)
        matw1 = Matrix([["x_{13}"],["x_{23}"],["\\vdots"],["x_{n3}"]]).next_to(matv,RIGHT)
        matv1 = Matrix([["x_{12}"],["x_{22}"],["\\vdots"],["x_{n2}"]]).move_to(LEFT)
        matp1 = MathTex("\dots").next_to(matw,RIGHT)
        matz1 = Matrix([["x_{1p}"],["x_{2p}"],["\\vdots"],["x_{np}"]]).next_to(matp,RIGHT)

        matv1.add(SurroundingRectangle(matv1.get_columns()[0]))
        matx1.add(SurroundingRectangle(matx1.get_columns()[0]))
        matw1.add(SurroundingRectangle(matw1.get_columns()[0]))
        matz1.add(SurroundingRectangle(matz1.get_columns()[0]))
        #matv = Matrix[[Matrix([["v_{x_1}"],["v_{x_2}"],["v_{x_3}"]]),Matrix([["v_{y_1}"],["v_{y_2}"],["v_{y_3}"]]),Matrix([["v_{z_1}"],["v_{z_2}"],["v_{z_3}"]])]]

        brv = Brace(matv).set_color(YELLOW)    
        brx = Brace(matx).set_color(YELLOW)    
        brz = Brace(matz).set_color(YELLOW) 
        brw = Brace(matw).set_color(YELLOW)  

        numv = Text("16%").next_to(brv,DOWN).set_color(YELLOW)  
        numx = Text("66%").next_to(brx,DOWN).set_color(YELLOW)
        numw = Text("8%").next_to(brw,DOWN).set_color(YELLOW)     
        numz = Text("0.1%").next_to(brz,DOWN).set_color(YELLOW)     

        self.add(matv,matx,matz,matp,matw,titulo)
        self.wait(16)
        self.play(FadeIn(brx),FadeIn(numx),FadeIn(brv),FadeIn(numv),FadeIn(brw),FadeIn(numw),FadeIn(brz),FadeIn(numz))
        self.wait(4)


class last(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(
            x_range=(-6,6),
            y_range=(-6,6),
            z_range=(-6,6),
            x_length=6,
            y_length=6,
            z_length=6
        )
        labx = axes.get_x_axis_label(Tex("X"))
        laby = axes.get_y_axis_label(Tex("Y"))
        labz = axes.get_z_axis_label(Tex("Z"))
        self.set_camera_orientation(phi=45*DEGREES,theta=-45*DEGREES,distance=8)

        dots = VGroup(*[Dot3D(point=axes.coords_to_point(p.x1,p.x2,p.x3), radius=0.05, color=YELLOW) for p in X2])
        dots2 = VGroup(*[Dot3D(point=axes.coords_to_point(p.x1_n,p.x2_n,p.x3_n), radius=0.05, color=YELLOW) for p in X3]  ) 

        func1 = ParametricFunction( lambda u: axes.c2p(u, u*(V[1][0]/V[0,0]), u*(V[2][0]/V[0][0])), color=RED , t_range=[-3,3])
        func2 = ParametricFunction( lambda u: axes.c2p(u, u*(V[1][1]/V[0,1]), u*(V[2][1]/V[0][1])), color=BLUE , t_range=[-3,3])
        func3 = ParametricFunction( lambda u: axes.c2p(u, u*(V[1][2]/V[0,2]), u*(V[2][2]/V[0][2])), color=GREEN , t_range=[-1,1])
        
        self.add(axes,labx,dots,laby,labz)
        self.wait(2)
        self.play(Transform(dots,dots2))
        self.wait(1)
        self.play(GrowFromCenter(func1),GrowFromCenter(func2),GrowFromCenter(func3))
        self.wait(2)
        self.move_camera(phi=0*DEGREES,theta=-90*DEGREES)
        self.wait(3)
        self.move_camera(phi=11.46*DEGREES,theta=-62*DEGREES,distance=4)
        self.wait(3)
        self.move_camera(phi=11.46*DEGREES,theta=-111.15*DEGREES)
        self.wait(2)
        self.move_camera(phi=45*DEGREES,theta=-45*DEGREES)
        self.wait(3)
        self.play(FadeOut(func3))
        self.begin_ambient_camera_rotation(rate=0.5)
        self.wait(5)
        self.stop_ambient_camera_rotation()
        #self.wait(2)



class thumbnail(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(
            x_range=(-6,6),
            y_range=(-6,6),
            z_range=(-6,6),
            x_length=6,
            y_length=6,
            z_length=6
        ).move_to(RIGHT*3.5+DOWN*0.6)
        labx = axes.get_x_axis_label(Tex("X"))
        laby = axes.get_y_axis_label(Tex("Y"))
        labz = axes.get_z_axis_label(Tex("Z"))
        self.set_camera_orientation(phi=0*DEGREES,theta=-90*DEGREES,distance=8)

        dots = VGroup(*[Dot3D(point=axes.coords_to_point(p.x1,p.x2,p.x3), radius=0.05, color=YELLOW) for p in X2])
        dots2 = VGroup(*[Dot3D(point=axes.coords_to_point(p.x1_n,p.x2_n,p.x3_n), radius=0.05, color=YELLOW) for p in X3]  ) 

        func1 = ParametricFunction( lambda u: axes.c2p(u, u*(V[1][0]/V[0,0]), u*(V[2][0]/V[0][0])), color=RED , t_range=[-3,3])
        func2 = ParametricFunction( lambda u: axes.c2p(u, u*(V[1][1]/V[0,1]), u*(V[2][1]/V[0][1])), color=BLUE , t_range=[-3,3])
        func3 = ParametricFunction( lambda u: axes.c2p(u, u*(V[1][2]/V[0,2]), u*(V[2][2]/V[0][2])), color=GREEN , t_range=[-1,1])

        m1 = Matrix([["x_{11}","x_{12} ","\dots","x_{1p} "],
        ["x_{21}","x_{22} ","\dots","x_{2p} "],
        ["x_{31}","x_{32} ","\dots","x_{3p} "],
        ["x_{41}","x_{42} ","\dots","x_{4p} "],
        ["\\vdots","\\vdots","\\ddots","\\vdots"],
        ["x_{n1}","x_{n2} ","\dots","x_{np} "]]).move_to(LEFT*5+DOWN).scale(0.6)

        m2 = Matrix([["z_{11}","z_{12}"],
        ["z_{21}","z_{22}"],
        ["z_{31}","z_{32}"],
        ["\\vdots","\\vdots"],
        ["z_{n1}","z_{n2}"]]).move_to(LEFT*1.5+DOWN).scale(0.6)

        text1 = MathTex("\\equiv").next_to(m2,LEFT).scale(1.2)

        titulo = Tex("Análisis por Componentes",font_size=80,color=YELLOW).to_edge(UP)
        titulo2 = Tex("Principlaes",font_size=80,color=YELLOW).next_to(titulo,DOWN)


        self.add(dots2,func1,func2,func3,axes,labx,laby,labz,m1,m2,text1,titulo,titulo2)