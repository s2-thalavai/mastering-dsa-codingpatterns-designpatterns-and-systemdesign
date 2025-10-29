# Multithreading and Concurrency

## 1. Core Concepts

| Concept                         | Description                                                                                 | Example                                              |
| ------------------------------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| **Thread**                      | Smallest unit of CPU execution within a process. Shares memory with other threads.          | `Thread` class in Java, `threading.Thread` in Python |
| **Concurrency**                 | Multiple tasks *making progress* at once — not necessarily simultaneously.                  | Switching between network I/O and CPU work           |
| **Parallelism**                 | Multiple tasks *executing simultaneously* — true simultaneous execution on multi-core CPUs. | Running matrix computations on 8 cores               |
| **Synchronization**             | Coordination between threads to safely access shared resources.                             | Locks, mutexes, semaphores                           |
| **Asynchronous / Non-blocking** | Uses event loops and callbacks rather than blocking threads.                                | `async/await` in JS or Python                        |


## 2. Concurrency vs Multithreading

| Term                 | Nature               | Typical Use                                             |
| -------------------- | -------------------- | ------------------------------------------------------- |
| **Concurrency**      | Structure            | Design pattern to manage multiple tasks effectively     |
| **Multithreading**   | Execution            | Low-level mechanism for true parallel execution         |
| **Asynchronous I/O** | Logical concurrency  | Efficient I/O (not CPU-bound)                           |
| **Multiprocessing**  | Physical parallelism | Bypass Global Interpreter Lock (GIL) or CPU-heavy tasks |

## 3. How Major Modern Languages Handle It

## ☕ Java

- Threads are first-class citizens.

- Executors: thread pools for managing parallel tasks.

- ForkJoinPool / Parallel Streams: high-level concurrency.

- CompletableFuture & Virtual Threads (Java 21):

    - Virtual Threads = lightweight threads (Project Loom).
    
    - Makes asynchronous programming simpler and scalable.

✅ Modern Java uses virtual threads for massive concurrency with low overhead.

## 🕸️ JavaScript (Node.js)

- Single-threaded, event-driven architecture.

- Concurrency via Event Loop (non-blocking I/O).

- Worker Threads for CPU-intensive work.

- Promises / async-await syntax for structured concurrency.

✅ Ideal for I/O-heavy apps, microservices, real-time APIs.


## 🐍 Python

- **Threads:** via threading module (but limited by the GIL)

- **Async:** via asyncio (event loop, async/await)

- **Multiprocessing:** via multiprocessing for CPU-bound tasks

**Framework examples:
**
- FastAPI uses asyncio for high concurrency.

- Celery for parallel background jobs.

✅ Use async I/O for network-bound, multiprocessing for CPU-bound tasks.

## 💨 Go (Golang)

- Goroutines: lightweight threads managed by Go runtime.

- Channels: safe communication between goroutines.

- Scheduler: multiplexes goroutines over OS threads.

✅ Built-in concurrency model — simple, scalable, efficient.

## 4. Patterns & Models

| Pattern                 | Description                               | Example                           |
| ----------------------- | ----------------------------------------- | --------------------------------- |
| **Thread Pool**         | Reuses threads to avoid creation overhead | Java ExecutorService              |
| **Actor Model**         | Isolated units communicate via messages   | Akka (JVM), Orleans (.NET)        |
| **Reactor Pattern**     | Event-driven async I/O                    | Node.js, Spring WebFlux           |
| **Pipeline / Dataflow** | Parallel processing in stages             | Go channels, .NET Dataflow        |
| **Fork/Join**           | Divide tasks recursively                  | Java ForkJoinPool, C++ std::async |

## 5. Common Issues

| Problem             | Description                                 | Mitigation                   |
| ------------------- | ------------------------------------------- | ---------------------------- |
| **Race condition**  | Two threads access shared data concurrently | Use locks, atomic variables  |
| **Deadlock**        | Two threads wait on each other’s locks      | Lock ordering, timeout locks |
| **Starvation**      | One thread never gets CPU time              | Fair scheduling              |
| **Thread overhead** | Too many threads waste resources            | Use async or thread pools    |

## 6. Modern Trends (2025)

- Structured concurrency (Go, Kotlin, Java Loom): ensures predictable lifetime of concurrent tasks.

- Virtual threads (Java 21+): millions of concurrent tasks.

- Async-first frameworks (FastAPI, Next.js API routes).

- Serverless parallelism (AWS Lambda, Azure Functions).

- Actor-based distributed concurrency (Orleans, Akka).

## 7. When to Use What

| Task Type                 | Best Model                        | Language Example              |
| ------------------------- | --------------------------------- | ----------------------------- |
| CPU-bound computation     | Multithreading or multiprocessing | Java, Go, C++                 |
| I/O-bound tasks           | Async / Event Loop                | Node.js, Python (asyncio)     |
| Real-time processing      | Actor / Reactive                  | Akka, Spring WebFlux          |
| High throughput APIs      | Virtual threads / async I/O       | Java Loom, FastAPI            |
| Distributed microservices | Message queues, Actors            | Kafka, RabbitMQ, Akka Cluster |

---

## 1. concurrency in action across Python, Java, and Go:

> “Fetch data from multiple URLs concurrently and measure total execution time.”

We’ll use **3 different concurrency models — async I/O, virtual threads, and goroutines** — so you can feel how each modern language approaches concurrency.

## 🐍 1. Python — asyncio (Asynchronous I/O Model)

```python
import asyncio
import aiohttp
import time

urls = [
    "https://example.com",
    "https://httpbin.org/get",
    "https://api.github.com"
]

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    start = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        print(f"Fetched {len(results)} URLs in {time.time() - start:.2f}s")

asyncio.run(main())
```

🧠 Concepts Used:

- Event loop (asyncio.run)

- Non-blocking I/O (aiohttp)

- Coroutines (async def)

- await ensures cooperative multitasking.

✅ Best For: network-heavy workloads (APIs, web crawlers).
⚠️ Not truly parallel for CPU-heavy tasks (GIL).

## ☕ 2. Java — Virtual Threads (Java 21+)

```java
import java.net.URI;
import java.net.http.*;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.*;

public class VirtualThreadExample {
    public static void main(String[] args) throws Exception {
        var client = HttpClient.newHttpClient();
        var urls = List.of(
            "https://example.com",
            "https://httpbin.org/get",
            "https://api.github.com"
        );

        var start = System.currentTimeMillis();
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var futures = urls.stream()
                .map(url -> executor.submit(() -> client.send(
                        HttpRequest.newBuilder(URI.create(url))
                            .timeout(Duration.ofSeconds(10))
                            .build(),
                        HttpResponse.BodyHandlers.ofString()
                )))
                .toList();

            for (var future : futures) {
                System.out.println("Fetched " + future.get().uri());
            }
        }
        System.out.println("Completed in " +
            (System.currentTimeMillis() - start) + " ms");
    }
}
```

🧠 Concepts Used:

- Virtual Threads (Project Loom): ultra-lightweight, millions possible.

- No need for async/await; just normal blocking code.

- Java runtime handles scheduling automatically.

✅ Best For: scalable APIs, microservices, parallel tasks.
💪 True concurrency, not limited like Python’s GIL.

## 🦩 3. Go — Goroutines and Channels

```go
package main

import (
    "fmt"
    "io"
    "net/http"
    "sync"
    "time"
)

func fetch(url string, wg *sync.WaitGroup) {
    defer wg.Done()
    resp, err := http.Get(url)
    if err != nil {
        fmt.Println("Error:", err)
        return
    }
    defer resp.Body.Close()
    io.ReadAll(resp.Body) // just to consume
    fmt.Println("Fetched:", url)
}

func main() {
    urls := []string{
        "https://example.com",
        "https://httpbin.org/get",
        "https://api.github.com",
    }

    start := time.Now()
    var wg sync.WaitGroup
    for _, url := range urls {
        wg.Add(1)
        go fetch(url, &wg)
    }
    wg.Wait()
    fmt.Printf("Completed in %.2fs\n", time.Since(start).Seconds())
}
```

🧠 Concepts Used:

- go keyword spawns lightweight goroutines.

- WaitGroup synchronizes them.

- Native concurrency, extremely efficient.

✅ Best For: concurrent network or compute workloads.
🚀 Can handle 100k+ goroutines easily.

## Performance & Model Comparison

| Language    | Model           | Parallelism | Complexity | Typical Concurrency | Strength                      |
| ----------- | --------------- | ----------- | ---------- | ------------------- | ----------------------------- |
| **Python**  | Async I/O       | Cooperative | Moderate   | 1K+ tasks           | Simplicity for I/O            |
| **Java 21** | Virtual Threads | True        | Low        | Millions            | Easy async replacement        |
| **Go**      | Goroutines      | True        | Very Low   | 100K+               | Lightweight, simple, scalable |

## Summary Visualization

| Task Type                | Best Language    | Why                                        |
| ------------------------ | ---------------- | ------------------------------------------ |
| I/O-bound microservices  | Python (FastAPI) | Async + simple syntax                      |
| Enterprise microservices | Java 21          | Virtual threads scale + stability          |
| Cloud-native systems     | Go               | Fast, minimal memory, concurrency built-in |
| Compute-heavy            | Java / Go        | True parallelism                           |
| Event-driven real-time   | Node.js / Go     | Non-blocking I/O, goroutines               |

---

## 3. move from I/O concurrency → to CPU-bound parallelism.

We’ll use the same task in all languages:

> “Find all prime numbers up to 1,000,000 using multiple workers concurrently.”

This will show how each language handles real parallel execution (CPU-heavy tasks).

### 🧮 Problem Definition

> Find primes ≤ 1,000,000 using concurrent workers.

Each worker checks a slice of the range.

## 🐍 1. Python — Multiprocessing (True Parallelism)

🧠 Note: Python threads don’t run CPU-heavy code truly in parallel (because of the GIL),
so we use multiprocessing — each process runs on its own CPU core.

```python
from multiprocessing import Pool, cpu_count
import math, time

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def count_primes_in_range(args):
    start, end = args
    return sum(1 for n in range(start, end) if is_prime(n))

if __name__ == "__main__":
    N = 1_000_000
    workers = cpu_count()
    chunk_size = N // workers
    ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(workers)]

    start = time.time()
    with Pool(workers) as p:
        total = sum(p.map(count_primes_in_range, ranges))
    print(f"Found {total} primes in {time.time() - start:.2f}s using {workers} processes")
```

✅ True parallelism (each worker = independent process)
⚠️ Overhead from inter-process communication (IPC).

## ☕ 2. Java — Virtual Threads or Fixed Thread Pool

🧠 Virtual Threads are more efficient for blocking tasks; for CPU tasks, use a fixed pool tied to CPU cores.

```java
import java.util.concurrent.*;
import java.util.stream.IntStream;

public class ParallelPrimes {
    static boolean isPrime(int n) {
        if (n < 2) return false;
        for (int i = 2; i * i <= n; i++)
            if (n % i == 0) return false;
        return true;
    }

    public static void main(String[] args) throws InterruptedException, ExecutionException {
        int N = 1_000_000;
        int cores = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(cores);

        int chunk = N / cores;
        var tasks = IntStream.range(0, cores)
                .mapToObj(i -> {
                    int start = i * chunk;
                    int end = (i + 1) * chunk;
                    return executor.submit(() -> {
                        int count = 0;
                        for (int n = start; n < end; n++)
                            if (isPrime(n)) count++;
                        return count;
                    });
                }).toList();

        long start = System.currentTimeMillis();
        int total = 0;
        for (var f : tasks) total += f.get();
        long duration = System.currentTimeMillis() - start;

        executor.shutdown();
        System.out.printf("Found %d primes in %d ms using %d threads%n", total, duration, cores);
    }
}
```

✅ True multithreading
💪 Scales with available CPU cores
🧠 Java optimizes thread scheduling at OS level.

## 🦩 3. Go — Goroutines + Channels (Simple and Fast)

🧠 Goroutines are very lightweight, and Go’s scheduler automatically spreads them across cores.

```go
package main

import (
    "fmt"
    "math"
    "runtime"
    "sync"
    "time"
)

func isPrime(n int) bool {
    if n < 2 {
        return false
    }
    sqrt := int(math.Sqrt(float64(n)))
    for i := 2; i <= sqrt; i++ {
        if n%i == 0 {
            return false
        }
    }
    return true
}

func countPrimes(start, end int, wg *sync.WaitGroup, result chan<- int) {
    defer wg.Done()
    count := 0
    for n := start; n < end; n++ {
        if isPrime(n) {
            count++
        }
    }
    result <- count
}

func main() {
    N := 1000000
    cores := runtime.NumCPU()
    runtime.GOMAXPROCS(cores)

    chunk := N / cores
    result := make(chan int, cores)
    var wg sync.WaitGroup

    start := time.Now()
    for i := 0; i < cores; i++ {
        wg.Add(1)
        go countPrimes(i*chunk, (i+1)*chunk, &wg, result)
    }

    go func() {
        wg.Wait()
        close(result)
    }()

    total := 0
    for count := range result {
        total += count
    }

    fmt.Printf("Found %d primes in %.2fs using %d cores\n",
        total, time.Since(start).Seconds(), cores)
}
```

✅ True parallelism with goroutines on multiple OS threads
💨 Lightweight: thousands of goroutines possible
🧩 Synchronization via channels

## Performance Expectation (Rough Benchmarks)

| Language   | Parallel Model  | Real Parallelism | Expected Runtime (1M primes, 8-core) |
| ---------- | --------------- | ---------------- | ------------------------------------ |
| **Python** | Multiprocessing | ✅ Yes            | ~3–5s                                |
| **Java**   | Thread Pool     | ✅ Yes            | ~2–4s                                |
| **Go**     | Goroutines      | ✅ Yes            | ~1.5–3s                              |

(Depends on CPU, memory, and implementation)

## Key Takeaways

| Language   | Concurrency Model      | CPU Parallelism | Simplicity     | Ideal Use                       |
| ---------- | ---------------------- | --------------- | -------------- | ------------------------------- |
| **Python** | Multiprocessing        | ✅               | ⚠️ Moderate    | Data science, background jobs   |
| **Java**   | Thread Pool / ForkJoin | ✅               | ⚙️ Medium      | Enterprise compute workloads    |
| **Go**     | Goroutines             | ✅✅              | 🧩 Very simple | Cloud-native concurrent systems |

---

## local concurrency → to distributed parallelism

moving from local concurrency → to distributed parallelism — where multiple nodes (servers) work together to 
solve CPU-intensive or data-intensive problems in parallel.

Let’s keep the same conceptual problem — “count primes up to N” — and scale it across a cluster using modern distributed concurrency models.

### 🌍 1. The Shift: Concurrency → Distributed Parallelism

| Concept            | Local (Single Machine)          | Distributed (Cluster)                            |
| ------------------ | ------------------------------- | ------------------------------------------------ |
| **Parallel Units** | Threads / Processes             | Nodes / Containers                               |
| **Communication**  | Shared memory / message passing | Network (HTTP, gRPC, MQ, Kafka)                  |
| **Scheduling**     | OS / language runtime           | Distributed orchestrator (K8s, Ray, Spark, Dask) |
| **Goal**           | Maximize CPU core usage         | Maximize all machines’ cores                     |

### 🚀 2. Three Distributed Approaches

We’ll cover:

- Python (Ray / Dask)

- Java (Akka / gRPC microservices)

- Go (Worker Pool + Message Queue)

## 🐍 A. Python — Using Ray (Distributed Task Execution)

Ray provides a Python-native distributed execution framework
→ think of it as “multiprocessing across many machines”.

```python
import ray
import math, time

# Start Ray cluster (or connect to existing one)
ray.init(address="auto")  # or ray.init() for local

@ray.remote
def count_primes(start, end):
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0: return False
        return True
    return sum(1 for n in range(start, end) if is_prime(n))

if __name__ == "__main__":
    N = 10_000_000
    num_workers = 20
    chunk = N // num_workers

    start = time.time()
    tasks = [count_primes.remote(i*chunk, (i+1)*chunk) for i in range(num_workers)]
    results = ray.get(tasks)
    total = sum(results)
    print(f"Found {total} primes in {time.time() - start:.2f}s using {num_workers} Ray workers")
```

✅ Distributed computation (each task may run on a different machine)
⚙️ Cluster orchestration built in
💡 Easy scaling — just add more Ray nodes.

## ☕ B. Java — Akka Cluster (Actor Model)

Akka lets you create distributed actors that process messages independently across nodes.
Perfect for massive parallelism + fault tolerance.

🧩 Cluster Architecture

- Master actor → divides work into chunks.

- Worker actors → deployed on multiple JVM nodes.

- Result collector → aggregates counts.

```scala
💻 Example (simplified)
// Scala code (Akka)
import akka.actor._

case class CountPrimes(start: Int, end: Int)
case class Result(count: Int)

class Worker extends Actor {
  def isPrime(n: Int) = n > 1 && (2 to math.sqrt(n).toInt).forall(n % _ != 0)

  def receive = {
    case CountPrimes(start, end) =>
      val count = (start until end).count(isPrime)
      sender() ! Result(count)
  }
}

class Master(numWorkers: Int, range: Int) extends Actor {
  val chunk = range / numWorkers
  val workers = (1 to numWorkers).map(_ => context.actorOf(Props[Worker]))
  var results = 0
  var pending = numWorkers

  override def preStart(): Unit = {
    workers.zipWithIndex.foreach { case (w, i) =>
      w ! CountPrimes(i * chunk, (i + 1) * chunk)
    }
  }

  def receive = {
    case Result(c) =>
      results += c
      pending -= 1
      if (pending == 0) {
        println(s"Total primes: $results")
        context.system.terminate()
      }
  }
}

object ClusterMain extends App {
  val system = ActorSystem("PrimeCluster")
  system.actorOf(Props(new Master(numWorkers = 8, range = 1_000_000)))
}
```

✅ Distributed actor system (via Akka Cluster)
💪 Resilient, message-driven, fault-tolerant
⚙️ Ideal for enterprise distributed workloads

## 🦩 C. Go — Worker Pool + Message Queue (RabbitMQ or NATS)

Go excels at simple, networked concurrency, so it’s perfect for distributed worker systems.

🧩 Architecture

Master Service sends tasks (ranges) via message queue.

Workers subscribe, compute prime counts, and publish results.

🧠 Master (Go + RabbitMQ)

```go
package main

import (
    "fmt"
    "log"
    "github.com/streadway/amqp"
)

func main() {
    conn, _ := amqp.Dial("amqp://guest:guest@localhost/")
    ch, _ := conn.Channel()
    q, _ := ch.QueueDeclare("tasks", false, false, false, false, nil)

    N := 1000000
    workers := 10
    chunk := N / workers

    for i := 0; i < workers; i++ {
        body := fmt.Sprintf("%d,%d", i*chunk, (i+1)*chunk)
        ch.Publish("", q.Name, false, false, amqp.Publishing{Body: []byte(body)})
    }

    fmt.Println("Distributed tasks to queue.")
}

🧩 Worker (runs on any node)
package main

import (
    "fmt"
    "log"
    "math"
    "strconv"
    "strings"
    "github.com/streadway/amqp"
)

func isPrime(n int) bool {
    if n < 2 { return false }
    sqrt := int(math.Sqrt(float64(n)))
    for i := 2; i <= sqrt; i++ {
        if n%i == 0 { return false }
    }
    return true
}

func main() {
    conn, _ := amqp.Dial("amqp://guest:guest@localhost/")
    ch, _ := conn.Channel()
    msgs, _ := ch.Consume("tasks", "", true, false, false, false, nil)

    for msg := range msgs {
        parts := strings.Split(string(msg.Body), ",")
        start, _ := strconv.Atoi(parts[0])
        end, _ := strconv.Atoi(parts[1])

        count := 0
        for n := start; n < end; n++ {
            if isPrime(n) { count++ }
        }
        fmt.Printf("Worker processed %d–%d, primes=%d\n", start, end, count)
    }
}
```

✅ Horizontally scalable — add more workers on any node
✅ Decoupled via RabbitMQ / NATS / Kafka
💪 Ideal for microservices or distributed compute grids.

## 4. Comparison — Distributed Concurrency

| Language            | Model                            | Communication       | Scaling            | Best For                       |
| ------------------- | -------------------------------- | ------------------- | ------------------ | ------------------------------ |
| **Python (Ray)**    | Task-based distributed execution | gRPC (Ray internal) | Linear             | ML, data pipelines, simulation |
| **Java (Akka)**     | Actor-based message passing      | Akka Cluster        | Linear + resilient | Event-driven microservices     |
| **Go (MQ Workers)** | Pub/Sub worker pool              | RabbitMQ / Kafka    | Linear             | Cloud microservices, pipelines |

## 🧠 6. Advanced Distributed Models (2025)

| Framework                      | Language       | Highlights                             |
| ------------------------------ | -------------- | -------------------------------------- |
| **Ray / Dask**                 | Python         | Cluster compute made easy              |
| **Akka / Quarkus / Micronaut** | Java           | Reactive microservices + actors        |
| **Temporal.io**                | Go / Java / TS | Workflow orchestration                 |
| **Celery + Redis**             | Python         | Task queue for async distributed jobs  |
| **Kafka Streams / Flink**      | JVM            | Streaming computation                  |
| **KEDA + RabbitMQ / NATS**     | Any            | Autoscaling distributed task executors |

## Summary

| Level   | Tool / Model                | Core Concept                | Example                 |
| ------- | --------------------------- | --------------------------- | ----------------------- |
| Local   | Threads / Goroutines        | Parallel tasks on 1 machine | CPU-bound concurrency   |
| Cluster | Actors / Workers / Tasks    | Distributed parallel tasks  | Multi-node compute grid |
| Global  | Message queues / Kubernetes | Event-driven autoscaling    | Cloud-native scaling    |

---

# Distributed concurrency in a Procure-to-Pay (P2P) example — end-to-end design (Spring Boot + FastAPI + Go)

Nice — here’s a full, practical blueprint you can use to build a distributed, concurrent P2P pipeline that handles invoice ingest 
→ OCR/document analysis → rule-engine validation → enrichment → posting/archival with horizontally-scalable microservices in Spring Boot (Java),
FastAPI (Python), and Go. I’ll include architecture, message flows, sample contracts, concise code snippets for each microservice, 
k8s deployment patterns, observability, scaling, error/Retry patterns and security.

## 1) High-level architecture (text diagram)

```java
[Client / ERP] --> API Gateway (Ingress) --> Invoice Ingest Service (FastAPI)
        |
        v
   Message Broker (Kafka/RabbitMQ)  <--->  OCR Service (Go)  <--->  Rule Engine (Spring Boot)
        |                                   |                     |
        v                                   v                     v
   Enrichment Service (Python/Go)      ML/Document DB         Result Collector / DB
        |
        v
   Accounting System (ERP) / Audit log
```

## Key ideas:

- **Async, event-driven:** Message broker as backbone (Kafka for high throughput / exactly-once-ish, RabbitMQ for simpler queues).

- **Microservice roles:** Ingest (HTTP), OCR (CPU + GPU friendly), Rule Engine (stateful business rules), Enricher (lookups), Result Collector (aggregate & persist).

- **Autoscale workers:** KEDA or HPA scale consumers based on queue lag / CPU.

## 2) Message model (JSON) — example contract

invoice.uploaded (produced by Invoice Ingest)

```json
{
  "invoice_id": "INV-2025-0001",
  "vendor_id": "V-420",
  "origin": "erp.systemA",
  "uploaded_at": "2025-10-29T07:12:31Z",
  "s3_uri": "s3://invoices-bucket/2025/10/INV-2025-0001.pdf",
  "metadata": {
    "amount": 2543.75,
    "currency": "USD",
    "invoice_date": "2025-10-25"
  }
}
```

invoice.ocr.completed

```json
{
  "invoice_id": "INV-2025-0001",
  "lines": [
    {"desc":"Widget","qty":2,"unit_price":1000.0}
  ],
  "supplier": {"name":"Acme", "tax_id":"12345"},
  "total": 2543.75,
  "ocr_confidence": 0.93
}
```

invoice.validation.result

```json
{
  "invoice_id":"INV-2025-0001",
  "status":"REJECTED",
  "violations":[
    {"rule_id":"R-13","message":"Tax mismatch","severity":"HIGH"}
  ],
  "timestamp":"2025-10-29T07:13:20Z"
}
```

Design note: small, immutable messages with clear versioning fields (schema_version) so services can evolve.

## 3) Message broker choices & topics/queues

- Kafka (recommended if high throughput & ordering matters)

    - Topics: invoice.uploaded, invoice.ocr, invoice.ocr.completed, invoice.validation.request, invoice.validation.result, invoice.enriched, invoice.audit
    
    - Partitioning: partition by invoice_id for ordering

- RabbitMQ / NATS (if simpler semantics)

    - Queues per consumer group, use dead-letter exchange for poison messages
 
## 4) Service responsibilities + short code sketches

A. Invoice Ingest — FastAPI (Python)

- Receives HTTP POST with file or S3 pointer

- Stores file to object-store and publishes invoice.uploaded to broker

- Light-weight, scale via replicas

Minimal snippet (publishing to Kafka with aiokafka):

```python
# ingest_service.py
from fastapi import FastAPI, UploadFile
from aiokafka import AIOKafkaProducer
import asyncio, json, uuid, datetime

app = FastAPI()
producer = None

@app.on_event("startup")
async def startup():
    global producer
    producer = AIOKafkaProducer(bootstrap_servers="kafka:9092")
    await producer.start()

@app.post("/upload")
async def upload(file: UploadFile):
    invoice_id = f"INV-{uuid.uuid4()}"
    # save to S3 (pseudo)
    s3_uri = f"s3://invoices/{invoice_id}.pdf"
    # publish event
    msg = {
        "invoice_id": invoice_id,
        "uploaded_at": datetime.datetime.utcnow().isoformat() + "Z",
        "s3_uri": s3_uri
    }
    await producer.send_and_wait("invoice.uploaded", json.dumps(msg).encode())
    return {"invoice_id": invoice_id}
```

**Concurrency:** FastAPI + Uvicorn workers (async) handle many concurrent uploads; use S3 presigned URLs for large files.

## B. OCR Service — Go (CPU-bound, parallel)

- Pulls invoice.uploaded tasks, downloads PDF from S3, runs OCR (Tesseract or cloud OCR), publishes invoice.ocr.completed.

- Use goroutines to parallelize pages; limit concurrency per worker by semaphore.

### Minimal (consumer outline):

```go
// consumer.go (pseudo)
func worker(msg InvoiceUploaded) {
    // download from S3
    // use OCR engine with worker-pool for pages
    // parse structured fields => ocrResult
    publish("invoice.ocr.completed", ocrResult)
}
```

**Concurrency:** each pod can run multiple goroutines; scale pods horizontally.

## C. Rule Engine — Spring Boot (Java)

- Subscribes to invoice.ocr.completed or invoice.validation.request

- Executes rules (Drools / custom engine). Could be stateful: hold past supplier flags, blacklists

- Emits invoice.validation.result and optionally triggers human review if severe.

Spring Boot + Kafka snippet (consumer + processing):

```java
// KafkaListener
@KafkaListener(topics = "invoice.ocr.completed")
public void onMessage(String payload) {
    InvoiceOcr ocr = objectMapper.readValue(payload, InvoiceOcr.class);
    ValidationResult result = ruleEngine.evaluate(ocr);
    kafkaTemplate.send("invoice.validation.result", objectMapper.writeValueAsString(result));
}
```

**Scaling:** use fixed thread pools sized to CPU cores to process rules concurrently. 
For complex rules consider a rule-service cluster with sticky routing when rule state matters.

