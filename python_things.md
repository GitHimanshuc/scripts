# Plotting

```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import numpy as np

print(plt.style.available)

max_val = 12
dx = 0.1**np.arange(0.1, max_val,0.1)
error = np.zeros(len(dx))
error = -((1+dx)**3-1.0)/dx + 3.0
log_errors = np.log10(np.abs(error))


mdx = 0.1**np.arange(0.1, max_val,0.1,dtype=np.longdouble)
merror = np.zeros(len(mdx),dtype=np.longdouble)
merror = -((1+mdx)**3-1.0)/mdx + 3.0
mlog_errors = np.log10(np.abs(merror))

ldx = 0.1**np.arange(0.1, max_val,0.1,dtype='f')
lerror = np.zeros(len(ldx),dtype='f')
lerror = -((1+ldx)**3-1.0)/ldx + 3.0
llog_errors = np.log10(np.abs(lerror))


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.log10(np.abs(ldx)), llog_errors,dashes=[1, 2],label="float32")
ax.plot(np.log10(np.abs(dx)), log_errors,label="float64")
ax.plot(np.log10(np.abs(mdx)), mlog_errors,dashes=[3, 2],label="float128")
ax.set_xlim(-0.1, -max_val)  # decreasing time
ax.set_xlabel('log(step size)')

ax.set_ylabel('log(error)')
ax.set_title('Error in first order finite difference approximation of the derivative of $x^3$')

ax.legend()
# ax.grid(True)
fig.savefig("/home/himanshu/Desktop/master_project/thesis/images/x^3_error_order1.png",bbox_inches='tight')
```


# Random
```python

print(np.finfo(np.float32))
print(np.finfo(np.float64))
print(np.finfo(np.longdouble))
```
# Subprocess
```python
process = subprocess.Popen(shlex.split("ls -l"),stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd="/home/himanshu/Desktop")
stdout, stderr = process.communicate() # To make sure that the python waits for the code to finish before moving on
```
