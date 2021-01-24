# SeaLion

![PyPI Deployment](https://github.com/KentoNishi/PythonPP/workflows/PyPI%20Deployment/badge.svg)
![Unit Tests](https://github.com/KentoNishi/PythonPP/workflows/Unit%20Tests/badge.svg)

A robust Java-style OOP system for Python, with support for statics, encapsulation, and inheritance.

[View on PyPI](https://pypi.org/project/pythonpp/)
/
Built by
[Kento Nishi](https://github.com/KentoNishi)
and
[Ronak Badhe](https://github.com/r2dev2bb8)

Python++ allows Python programmers to use object oriented programming principles in Python.


## Installation
The package is available on PyPI.
You can install the package with the following command:
```shell
pip install pythonpp
```

## Usage

### Importing the Library
You can import Python++ using a wildcard import statement.
```python
from pythonpp import *
```

### Class Declaration
Declare Python++ classes with the `@PythonPP` decorator.

```python
@PythonPP
class MyClass:
    pass # class code here
``` 

### Namespace and Scopes
Declare variables and methods for Python++ classes within `namespace`.

```python
@PythonPP
class MyClass:
    def namespace(public, private):
        pass # methods and variables here
```

Code within `namespace` has access to the following scopes:

| Scope | Description |
|:------|:------------|
| `public` | The public instance scope. |
| `private` | The private instance scope. |
| `public.static` | The public static scope. |
| `private.static` | The private static scope. |

### Static Initializers
Declare static initializers for Python++ classes using the `@staticinit` decorator.
Static initializers do not have access to instance variables and methods.
Static initializers cannot have input parameters.

```python
@PythonPP
class MyClass:
    def namespace(public, private):
        @staticinit
        def StaticInit():
            public.static.publicStaticVar = "Static variable (public)"
            private.static.privateStaticVar = "Static variable (private)"
```

Alternatively, static variables can be declared in the bare `namespace` **if the variable assignments are constant**. Using bare static variable declarations are **not recommended**.


### Constructors
Constructors can be declared using the `@constructor` decorator. Constructors can have input parameters.

```python
@PythonPP
class MyClass:
    def namespace(public, private):
        @constructor
        def Constructor(someValue):
            public.publicInstanceVar = "Instance variable (public)"
            public.userDefinedValue = someValue
```

### Method Declarations
Methods are declared using the `@method(scope)` decorator with the `public` and `private` scopes in `namespace`.

```python
@PythonPP
class MyClass:
    def namespace(public, private):
        @method(public)
        def publicMethod():
            pass # public instance method here
        
        @method(private)
        def privateMethod():
            pass # private instance method here
        
        @method(public.static)
        def publicStaticMethod():
            pass # public static method here
        
        @method(private.static)
        def privateStaticMethod():
            pass # private static method here
```

### Special Methods
Declare special built-in methods using the `@special` decorator.
```python
@PythonPP
class MyClass:
    def namespace(public, private):
        @special
        def __str__():
            return "Some string value"
```

### Inheritance
Classes can extend other classes using standard Python class inheritance.
```python
@PythonPP
class ParentClass:
    def namespace(public, private):
        @staticinit
        def StaticInit():
            public.static.staticVar = "Static variable"

        @constructor
        def Constructor(param):
            print("Parent constructor")
            public.param = param

@PythonPP
class ChildClass(ParentClass): # ChildClass extends ParentClass
    def namespace(public, private):
        @staticinit
        def StaticInit():
            ParentClass.staticinit() # Call parent static initializer
            public.static.staticVar2 = "Static variable 2"

        @constructor
        def Constructor(param):
            # Call parent constructor
            ParentClass.constructor(param)
```

## Quickstart Example
```python
from pythonpp import *

@PythonPP
class ParentClass:
    def namespace(public, private):
        @staticinit
        def StaticInit():
            public.static.publicStaticVar = "Public static variable"
            private.static.privateStaticVar = "Private static variable"

        @constructor
        def Constructor(parameter):
            private.privateVariable = parameter

@PythonPP
class ChildClass(ParentClass):
    def namespace(public, private):
        @staticinit
        def StaticInit():
            ParentClass.staticinit()

        @constructor
        def Constructor(parameter):
            ParentClass.constructor(parameter)
            public.publicVariable = "Public variable"
            private.privateVariable = "Private variable"
        
        @method(public)
        def getPrivateVariable():
            return private.privateVariable
        
        @method(public.static)
        def getPrivateStaticVar():
            return private.static.privateStaticVar

        @special
        def __str__():
            return "ChildClass object"
```
```python
print(ChildClass.publicStaticVar)
# > Private static variable
print(ChildClass.getPrivateStaticVar())
# > Private static variable

obj = ChildClass("Parameter value")
print(obj)
# > ChildClass object
print(obj.publicVariable)
# > Public variable
print(obj.getPrivateVariable())
# > Parameter value
try:
    obj.privateVariable # results in an error
except Exception as e:
    print(e)
# > 'ChildClass' object has no attribute 'privateVariable'
```

## Full Example
You can view the full example Jupyter notebook [here](https://github.com/r2dev2bb8/PythonPP/blob/master/examples/example.ipynb).
