(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects i d a h c e j l f g k)
(:init 
(handempty)
(ontable i)
(ontable d)
(ontable a)
(ontable h)
(ontable c)
(ontable e)
(ontable j)
(ontable l)
(ontable f)
(ontable g)
(ontable k)
(clear i)
(clear d)
(clear a)
(clear h)
(clear c)
(clear e)
(clear j)
(clear l)
(clear f)
(clear g)
(clear k)
)
(:goal
(and
(on i d)
(on d a)
(on a h)
(on h c)
(on c e)
(on e j)
(on j l)
(on l f)
(on f g)
(on g k)
)))