---
title: Python의 변수는 어떻게 관리되는가?
date: 2023-12-13 20:29:00 +0800
categories: [Language, Python]
tags: [python, variables, objects, object-oriented programming, memory management, variable formation, programming language, object model, dynamic typing, programming basics]
use_math: true
---
# Python
다른 언어들은 변수를 선언할 때 타입을 함께 명시해 준다. 하지만 파이썬은 변수 타입과 메모리 할당을 따로 해주지 않아도 된다는 점에서 코딩 입문자가 접하기 쉬운 언어인 것 같다. 하지만 파이썬은 다른 언어와 어떻게 차이가 있길래 변수 선언 방식이 다른걸까?

해당 주제는 Ned Batchelder의 (PyCon 2015 발표)[https://youtube.com/watch?v=_AEJHKGk9ns]를 참고하여 작성했다. Ned의 설명이 너무 깔끔해서 내용을 정리하고자 작성한다.

## Names and Values
Python에서는 `name is a reference to a value`라고 한다. 이게 무슨 뜻이냐 하면,

```python
x = 23
```
의 경우 변수 `x`가 `23`을 refer 한다고 한다. 나중에 이 x의 값을 아용하게 되면,

```python
print(x+2) #25
```

가 된다. 이를 쉽게 이해하기 위해서 아래와 같이 `이름`을 회색 태그로 나타내고, `화살표`로 값을 가리키는 그림을 그려볼 수 있다.

<img src="{{page.img_pth}}pyob1.svg">

이 과정을 Python에서는 `assignmnet`라고 한다. Assignment는 하나의 값에 대해서 여러번 다른 이름으로 수행 될 수 있다.

```python
x = 23
y = x
```
<img src="{{page.img_pth}}pyob2.svg">

이 경우, `x`와 `y`는 같은 값을 "refer"하고 있다. `x`와 `y`는 값의 고유 이름이 아니라 같은 값을 가리키고 있는 상태 라고 이해하면 된다. 다른말로, 이 두 이름은 서로 관련되어 있지는 않다. 다른 말로, 한 이름을 다른 값으로 재할당 해주면 다른 이름에 영향을 주지 않는다.

```python
x = 23
y = x
x = 12
```
<img src="{{page.img_pth}}pyob3.svg">

여기서 `y = x`는 "y가 x와 동일하다" 보다 "x와 y가 같은 값을 가리키고 있다"가 더 정확한 표현인것 같다.

Python에서의 값은 자신을 가리키고 있는 화살표, 즉 이름이 없을 때 메모리 해제를 시켜준다. 이를 garbage collection이라고 한다. Reference counting의 수가 0이 될 때, 더이상 쓰이지 않는 값이라고 판단하여 값을 메모리에서 지워주게 되는데, 나중에 자세하게 gc에 대해 다뤄보도록 하겠다.

## Assignment
Python에서 값을 이름에 assign 해줄 때, 값을 절대로 복사해주지 않고, 새로운 값을 절대 만들어 내지 않는다. 방금 전 예시를 보면 x와 y, 두 변수가 생성된 것처럼 보이지만, `y = x`를 해주었을 때 값은 하나고, 이름표가 두개 만들어 진 것이다. 일반 정수형 말고 리스트와 같이 복잡한 데이터를 보자.

```python
nums = [1, 2, 3]
```
<img src="{{page.img_pth}}pyob4.svg">

```python
nums = [1, 2, 3]
tri = nums
```
<img src="{{page.img_pth}}pyob5.svg">

전의 예시와 같이 하나의 list데이터가 있고, 두개의 이름표 (nums, tri)가 해당 리스트를 가리키고 있다. Python의 값은 dict, set, user-defined objects와 같이 가변적인 `mutable`값과, 불가변적인 int, strings, tuples값은 `immutable`값이라고 한다.

Mutable 객체는 값 자체를 in-place로 조작할 수 있는 method가 있다는 것이 특징이고, immutable 객체는 값이 절대로 변하지 않는다는 특징이 있다. 하나 예시를 들어보자.

```python
x = 1
x = x + 1
```
과 같은 코드에서 2라는 값이 새로 만들어지고, x라는 이름표는 2라는 데이터에 재할당 해주게 되는 것이다.

반대로 mutable 객체는 값을 바로 조작 할 수 있는 메소드를 사용 할 수 있다.
```python
nums = [1, 2, 3]
nums.append(4)
```
<img src="{{page.img_pth}}pyob6.svg">

<img src="{{page.img_pth}}pyob7.svg">

이렇게 다른 방식으로 값을 조작하는 것을 ***"rebinding the value VS. mutating the value"*** 라고 한다. 이 사실을 알면 다음과 같이 신기한 조작도 가능하다!

```python
nums = [1, 2, 3]
tri = nums
nums.append(4)

print(tri)      # [1, 2, 3, 4]
```

분명 nums를 조작했지만, nums와 tri는 같은 데이터 (이 경우 list)를 가리키고 있기 때문에 tri의 출력 값도 변하게 되는 것이다! 해당 기능은 버그가 아니기 때문에 python을 사용할 때 생각하면서 사용해야 한다.

## Diversity
Python에서의 list, dictionary와 같은 객체 안에 들어있는 정보도 모두 `reference`다. 따라서 전에 그렸었던 그림을 다음과 같이 해석할 수 있다.

<img src="{{page.img_pth}}pyob8.svg">

이 사실을 알면, 다음과 같은 작업도 모두 `referencing`이라고 볼 수 있다.

```python
# 왼쪽은 reference, 오른쪽은 value
my_obj.attr = 23
my_dict[key] = 24
my_list[index] = 25
my_obj.attr[key][index].attr = "etc, etc"

# X 라는 이름에 assignment
X = ...
for X in ...
[... for X in ...]
(... for X in ...)
{... for X in ...}
class X(...):
def X(...):
def fn(X): ... ; fn(12)
with ... as X:
except ... as X:
import X
from ... import X
import ... as X
from ... import ... as X
```

이러한 원리로 인해 python의 함수는 call by value, call by reference가 아닌 `call by assignment`인 것이다.

```python
def immutableFunction(val):
    val = val + 100

def mutableFunction(li, val):
    li.append(val)

num = 1
nums = [1, 2, 3]

immutableFunction(num)
mutableFunction(nums, 100)

print(f"num: {num}\nnums: {nums}")
# num: 1
# nums: [1, 2, 3, 100]
```
위 코드를 차근차근 보자.
- `immutableFunction`에서의 `val`은 local 변수
- `num`은 `val`에 assign 됨
- `val` 은 `val + 1`인 101로 재할당을 함
- local variable인 `val`변수는 함수가 종료되고 reference(화살표)를 삭제
- gc에 의해 `val`이 가리키고 있던 101의 값은 삭제
- 만약 `return val`을 사용하게 되면 reference가 남아 있기 때문에 101이라는 값을 살릴 수 있음

- `mutableFunction`에서의 `li`도 위와 같이 작동함
- `li`가 reference 하고 있는 리스트를 조작함
- `li`와 함수 밖의 `nums`는 같은 값을 가리키고 있기 때문에 값이 성공적으로 바뀜

mutableFunction이 조금 헷갈릴 수 있으니 그림으로 이해해 보자.

<img src="{{page.img_pth}}pyob9.svg">

1. Assignment never copies data.
2. Python is neither "Call By value" nor "Call By Reference", it's "Call by Assignment"! Epic! 
3. There is no way in python where a name can refer to another name. A name can only refer to values. Oh my!
