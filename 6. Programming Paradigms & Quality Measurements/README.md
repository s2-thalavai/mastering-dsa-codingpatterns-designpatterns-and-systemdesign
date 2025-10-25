# Programming Paradigms & Quality Measurements

1. Programming Paradigms
2. Quality Measurements

## 1. Programming Paradigms

Programming Paradigms are fundamental styles or approaches to programming — they define how you think about and structure your code to solve problems. 

Each paradigm provides its own model of computation, abstraction mechanisms, and design philosophy.

## Overview

| **Paradigm**              | **Core Idea**                                                                     | **Examples of Languages**     |
| ------------------------- | --------------------------------------------------------------------------------- | ----------------------------- |
| **Imperative**            | Tell the computer *how* to do something step by step.                             | C, Python, Java               |
| **Declarative**           | Tell the computer *what* you want done, not how to do it.                         | SQL, HTML, Prolog             |
| **Procedural**            | Organize code into reusable *procedures* or *functions*.                          | C, Pascal, Fortran            |
| **Object-Oriented (OOP)** | Model problems using *objects* (data + behavior).                                 | Java, Python, C++, C#         |
| **Functional**            | Treat computation as evaluation of *mathematical functions*; avoid mutable state. | Haskell, Scala, Lisp, Elixir  |
| **Logic**                 | Express computation in terms of *logical relations* and rules.                    | Prolog, Datalog               |
| **Event-Driven**          | Code responds to *events* (user input, messages, signals).                        | JavaScript, Node.js, C# (GUI) |
| **Concurrent / Parallel** | Focus on *multiple computations happening simultaneously*.                        | Go, Erlang, Java (threads)    |
| **Reactive**              | Build systems that *react to data streams and changes* over time.                 | RxJS, Akka, Kotlin Flow       |


## 1. Imperative Paradigm

- **Concept:** You write explicit instructions for the computer to change program state.

- **Key idea:** Sequence of commands.

- **Example:**

```python
x = 5
y = 10
sum = x + y
print(sum)
```

- Used in low-level programming and most general-purpose languages.

## 2. Procedural Paradigm

Extension of Imperative

- **Focus:** Break program into procedures (functions/subroutines).

- **Example:**

```c
int add(int a, int b) {
    return a + b;
}
printf("%d", add(5, 10));
```

- Promotes code reuse and modularity.

## 3. Object-Oriented Paradigm (OOP)

- **Core concepts:** Encapsulation, Inheritance, Polymorphism, Abstraction.

- **Structure:** Code organized around objects that represent real-world entities.

- **Example:**

```java
class Invoice {
    private double amount;
    Invoice(double amount) { this.amount = amount; }
    public double getAmount() { return amount; }
}
```

- Widely used in enterprise software and frameworks (Java, C#, Python).

## 4. Functional Paradigm

- **Core idea:** Functions are first-class citizens and should be pure (no side effects).

- **Avoids:** Mutable state and loops.

- **Example:**

```python
numbers = [1, 2, 3, 4]
squared = list(map(lambda x: x*x, numbers))
```

- Encourages immutability, recursion, and composability.

## 5. Declarative Paradigm

- Tell what to do, not how to do it.

- Common in database queries, configuration, and UI definitions.

- Example (SQL):

```sql
SELECT name FROM employees WHERE salary > 50000;
```

- Used in SQL, HTML, CSS, Prolog.

## 6. Logic Paradigm

- **Based on:** Formal logic and inference.

- You define facts and rules; the system deduces answers.

- **Example (Prolog):**

```prolog
parent(siva, kumar).
parent(kumar, arun).
grandparent(X, Y) :- parent(X, Z), parent(Z, Y).
```

- Used in AI, theorem proving, and expert systems.

## 7. Event-Driven Paradigm

- System reacts to events like user actions, messages, or signals.

- **Example (JavaScript):**

```javascript
button.addEventListener("click", () => alert("Clicked!"));
```

- Dominant in UI, games, and asynchronous systems.

## 8. Concurrent & Parallel Paradigm

- **Concurrent:** Multiple tasks appear to run simultaneously.

- **Parallel:** Multiple tasks actually run simultaneously on multiple cores.

- **Example (Go):**

```go
go task1()
go task2()
```

- Used in distributed systems, high-performance computing, and real-time apps.

## 9. Reactive Paradigm

- **Focus:** Data flows and propagation of change.

- **Example (RxJS):**

```javascript
from([1,2,3,4])
  .pipe(map(x => x * 2))
  .subscribe(console.log);
```

- Used in real-time UIs, IoT, streaming systems.


## Hybrid Languages

Most modern languages support multiple paradigms:

| Language           | Supported Paradigms                  |
| ------------------ | ------------------------------------ |
| **Python**         | OOP, Functional, Procedural          |
| **JavaScript**     | Functional, Event-driven, OOP        |
| **Java**           | OOP, Functional (from Java 8)        |
| **Scala / Kotlin** | OOP + Functional                     |
| **Go**             | Imperative + Concurrent              |
| **Rust**           | Imperative + Functional + Concurrent |

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/0acc47a3-cfd5-4a77-8eb3-efc77abfd80d" />

## Summary

| Paradigm                | Focus                  | Benefits                    | Common Use               |
| ----------------------- | ---------------------- | --------------------------- | ------------------------ |
| **Imperative**          | Step-by-step execution | Simple and direct           | System-level programming |
| **Procedural**          | Reusable procedures    | Modularity                  | Classic applications     |
| **OOP**                 | Real-world modeling    | Maintainability             | Enterprise software      |
| **Functional**          | Pure functions         | Predictability, testability | Data transformations     |
| **Declarative**         | Specify *what*         | Simplicity                  | Databases, configs       |
| **Logic**               | Knowledge rules        | Reasoning                   | AI, Expert systems       |
| **Event-Driven**        | Event response         | Interactivity               | GUIs, Web apps           |
| **Concurrent/Parallel** | Multi-tasking          | Performance                 | Cloud & HPC apps         |
| **Reactive**            | Streams & events       | Responsiveness              | Real-time systems        |

## 2. Quality Measurements
