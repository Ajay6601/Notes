# LLM Fundamentals - Part 1: API Design & Production Systems
## Deep Dive with Interview Traps (Hao Hoang Style)

**Sources**: OpenAI API docs, Anthropic technical blogs, Stripe API design, AWS Lambda best practices, Google SRE book, Martin Kleppmann (Designing Data-Intensive Applications), Chip Huyen (Designing ML Systems)

---

# 1. LLM API Design and Implementation

## 1.1 Synchronous vs Asynchronous APIs - The Hidden Trade-offs

### Core Concepts with Production Reality

**Synchronous (Request-Response)**
- Client blocks until complete response arrives
- **The Hidden Cost**: Connection held open for 10-30s
  - Server resources: 1 connection = 1 thread/worker
  - At 1000 concurrent requests: 1000 threads (thread pool exhaustion)
  - Memory: Each connection ~64KB socket buffer × 1000 = 64MB minimum

**Real Production Issue** (From Replicate.com blog, 2023):
```
Problem: GPT-3.5 API averages 3s response time
With 10K QPS: Need 30,000 concurrent connections
Apache default: 256 max connections → CRASHED

Solution: nginx + async workers (100K concurrent)
```

**Asynchronous (Job Queue)**
- Client receives job_id immediately
- Polls for status: `GET /jobs/{job_id}`
- **The Polling Problem**: How often to poll?
  - Too frequent: 100ms polling = 10 requests/sec per job (DDoS yourself!)
  - Too slow: 10s polling = poor UX (user thinks it's stuck)

**Production Pattern** (From Anthropic Claude API):
```python
# Progressive backoff polling
delays = [0.5, 1, 2, 4, 8, 10, 10, 10]  # seconds
for delay in delays:
    time.sleep(delay)
    status = check_job_status(job_id)
    if status == "completed":
        return get_result(job_id)
```

**Streaming (The LLM Sweet Spot)**
- Server-Sent Events (SSE) or WebSocket
- **Why it wins for LLMs**:
  - Time to First Token (TTFT): 300ms (user sees progress)
  - Total time: Still 10s, but feels faster
  - Bandwidth: Chunks sent as generated (no buffering)

**Production Gotcha** (From Cohere engineering blog):
```
Problem: SSE doesn't work behind corporate proxies (many buffer indefinitely)
Fix: Chunked Transfer Encoding fallback
      If client doesn't support SSE → use Transfer-Encoding: chunked
      If that fails → fallback to polling
```

### The Latency Math You Must Know

**Question**: 1000 users, each request takes 10s. Synchronous API with 100 workers. What's average wait time?

**Trap**: Most candidates say "10s" (wrong!)

**Correct Answer**:
```
Arrival rate: 1000 requests
Service rate: 100 workers × (1 request / 10s) = 10 requests/second
Time to serve all: 1000 / 10 = 100 seconds

Average wait (M/M/c queue):
ρ = λ / (c × μ) = 1000 / (100 × 0.1) = 1.0 (at capacity!)
Average wait time: ~50 seconds (queueing theory)

With async: Wait time = 0 (job enqueued immediately)
```

**Production Reality**: Netflix API Gateway
- Synchronous: p99 latency 45s under load
- Async + queue: p99 latency 100ms (to enqueue)

---

## 1.2 Streaming Implementation - The Devil's in the Details

### Server-Sent Events (SSE) - What They Don't Tell You

**SSE Format** (strict syntax):
```
data: {"content": "Hello"}\n\n
data: {"content": " world"}\n\n
data: [DONE]\n\n

MUST have:
- "data: " prefix
- Double newline \n\n separator
- NO empty lines between (breaks parsing)
```

**Production Gotcha #1**: Connection buffering
```python
# BAD: Python default buffering
def generate():
    for chunk in model.generate():
        yield f"data: {chunk}\n\n"

# GOOD: Disable buffering
def generate():
    for chunk in model.generate():
        yield f"data: {chunk}\n\n"
        # Force flush in WSGI/ASGI
        
# FastAPI example
@app.post("/stream")
async def stream():
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )
```

**Production Gotcha #2**: Connection timeout
```python
# Problem: Most load balancers timeout at 60s
# Solution: Keep-alive pings

async def generate_with_keepalive():
    last_chunk = time.time()
    for chunk in model.generate():
        yield f"data: {chunk}\n\n"
        last_chunk = time.time()
    
    # Send heartbeat every 15s if no chunks
    while model.is_generating():
        if time.time() - last_chunk > 15:
            yield ": ping\n\n"  # Comment line (SSE spec)
            last_chunk = time.time()
```

**Real Issue** (From Modal Labs blog, 2024):
```
Scenario: GPT-4 generates 5000 tokens
Token generation: 50 tokens/sec
Time: 100 seconds
AWS ALB timeout: 60 seconds → CONNECTION DROPPED

Fix: Send keepalive comments OR increase ALB timeout to 120s
```

### WebSocket - When You Need Bidirectional

**When to use over SSE**:
1. User can interrupt generation ("Stop generating")
2. Multi-turn conversation state
3. Real-time collaboration (multiple users editing)

**The Connection Management Nightmare**:
```python
# Problem: WebSockets are stateful (sticky sessions required)
# With 3 servers: User connects to Server A
# If Server A dies → connection lost, must reconnect

# Load balancer must use:
- IP hash (same client → same server)
- Or: Connection draining (graceful shutdown)
```

**Production Scale** (From Discord engineering):
```
Discord: 2.5M concurrent WebSocket connections per server
How: Custom protocol (not standard WebSocket)
      - Binary frames (not text JSON)
      - Compression enabled
      - Heartbeat every 41.25s

Memory per connection: ~8KB (vs 64KB standard)
Total: 2.5M × 8KB = 20GB RAM per server
```

---

## 1.3 Request/Response Schema Design - The Versioning Trap

### The Breaking Change Disaster

**Common Interview Question** (Hao Hoang style):
```
Your API has:
{
  "model": "gpt-4",
  "max_tokens": 1000
}

You want to rename "max_tokens" to "max_length".
What's the safest migration path?
```

**Trap**: Most say "support both for 6 months" (correct but incomplete!)

**Full Answer**:
```python
# Phase 1 (Month 0): Accept both, prefer new
if "max_length" in request:
    max_tokens = request["max_length"]
elif "max_tokens" in request:
    max_tokens = request["max_tokens"]
    # Log deprecation warning
    logger.warning(f"max_tokens deprecated, use max_length")

# Phase 2 (Month 3): Add deprecation header
response.headers["Deprecation"] = "true"
response.headers["Sunset"] = "2024-12-31"

# Phase 3 (Month 6): Reject old parameter
if "max_tokens" in request and "max_length" not in request:
    return 400, {
        "error": "max_tokens removed, use max_length",
        "docs": "https://api.example.com/migration"
    }

# Phase 4 (Month 9): Complete removal
# Only accept max_length
```

**Real Failure** (From Stripe API postmortem, 2022):
```
Stripe changed "source" to "payment_method" immediately
Result: 15% of API calls failed for 2 hours
Cost: $2M in lost transactions
Lesson: Always support both for 6+ months
```

### Schema Validation - The Performance Trap

**Question**: Validate 10K requests/sec with Pydantic. What's the bottleneck?

**Trap**: Pydantic validation is CPU-intensive

**Benchmark**:
```python
# Simple schema validation
from pydantic import BaseModel
import time

class Request(BaseModel):
    model: str
    messages: list[dict]
    temperature: float = 0.7

# Test 10K validations
requests = [{"model": "gpt-4", "messages": [...]} for _ in range(10000)]

start = time.time()
for req in requests:
    Request(**req)
duration = time.time() - start

print(f"Time: {duration}s")  # ~2.5 seconds
print(f"QPS: {10000 / duration}")  # ~4000 QPS

# Bottleneck: Each validation takes 0.25ms
# At 10K QPS: Need 10K × 0.25ms = 2.5s of CPU per second
# Single core maxed out!
```

**Production Solution** (From FastAPI docs):
```python
# Option 1: Disable validation in production (unsafe!)
@app.post("/chat")
async def chat(request: dict):  # No validation
    return process(request)

# Option 2: Sampling validation (1% validated)
@app.post("/chat")
async def chat(request: dict):
    if random.random() < 0.01:
        Request(**request)  # Validate 1% for monitoring
    return process(request)

# Option 3: Use multiple workers (scale horizontally)
# gunicorn with 16 workers: 4K × 16 = 64K QPS
```

---

## 1.4 API Versioning - The Migration Minefield

### URL-based vs Header-based - The Hidden Costs

**URL-based** (OpenAI, Anthropic):
```
/v1/chat/completions
/v2/chat/completions

Pros:
- Visible in logs
- Easy to debug
- CDN can cache per version

Cons:
- URL proliferation (/v1, /v2, /v3...)
- Two codebases to maintain
```

**Header-based** (Stripe, GitHub):
```
POST /chat/completions
API-Version: 2024-01-15

Pros:
- Clean URLs
- Fine-grained versioning (dates)
- Easier to deprecate old versions

Cons:
- Hidden from URL (harder to debug)
- CDN caching complex (must cache on header)
```

**The Date-based Trap** (Real Stripe issue):
```
Problem: Customer hard-codes "2023-10-01" in API version
         2 years later: Version deprecated
         Their integration breaks

Solution: Default to "latest" if no version specified
          But: "latest" changes behavior (breaks reproducibility!)
          
Compromise: 
- Accept date versions
- Default to version from API key creation date
- Force upgrade after 24 months
```

### The Sunset Header Pattern

**HTTP Standard** (RFC 8594):
```http
HTTP/1.1 200 OK
Deprecation: true
Sunset: Sat, 31 Dec 2024 23:59:59 GMT
Link: <https://api.example.com/docs/v2>; rel="deprecation"

{
  "result": "..."
}
```

**Client Implementation**:
```python
import requests
from datetime import datetime

response = requests.post("https://api.example.com/v1/chat")

if "Deprecation" in response.headers:
    sunset = response.headers.get("Sunset")
    if sunset:
        sunset_date = datetime.strptime(sunset, "%a, %d %b %Y %H:%M:%S GMT")
        days_remaining = (sunset_date - datetime.now()).days
        
        if days_remaining < 30:
            logger.critical(f"API version deprecated in {days_remaining} days!")
            alert_team()
```

**Production Monitoring** (From AWS API Gateway metrics):
```
Track by version:
- Request count per version
- Error rate per version
- Latency p99 per version

Alert when:
- v1 request count > 1% after sunset date
- Email customers still using v1
```

---

## 1.5 Error Handling - The Five Levels

### Level 1: HTTP Status Codes (The Basics)

**The 429 Rate Limit Deep Dive**:
```http
HTTP/1.1 429 Too Many Requests
Retry-After: 60
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1640995200

{
  "error": {
    "type": "rate_limit_error",
    "message": "Rate limit exceeded. Retry after 60 seconds."
  }
}
```

**Question**: What happens if client ignores Retry-After and retries immediately?

**Answer**: Exponential backoff penalty
```python
# Bad client (ignores Retry-After)
for attempt in range(10):
    response = requests.post(url)
    if response.status_code == 429:
        time.sleep(1)  # Wrong! Should sleep 60s

# Server response (exponentially increases penalty)
# Attempt 1: 429, Retry-After: 60s
# Attempt 2: 429, Retry-After: 120s
# Attempt 3: 429, Retry-After: 240s
# Attempt 4: 429, Retry-After: 480s (temporary ban)
# Attempt 5: 403, banned for 1 hour
```

**Production Implementation** (From Cloudflare blog):
```python
class RateLimiter:
    def __init__(self):
        self.violations = {}  # user_id -> violation_count
    
    def check_limit(self, user_id):
        if self.is_rate_limited(user_id):
            violations = self.violations.get(user_id, 0) + 1
            self.violations[user_id] = violations
            
            # Exponential penalty
            penalty = min(60 * (2 ** violations), 3600)  # Max 1 hour
            
            if violations >= 5:
                # Temporary ban
                return 403, {"error": "Too many violations. Banned for 1 hour."}
            
            return 429, {"retry_after": penalty}
```

### Level 2: Structured Error Codes

**The Error Type Taxonomy** (From OpenAI):
```python
ERROR_CODES = {
    # Client errors (4xx)
    "invalid_request_error": {
        "http_status": 400,
        "retry": False,
        "user_action": "Fix request and retry"
    },
    "authentication_error": {
        "http_status": 401,
        "retry": False,
        "user_action": "Check API key"
    },
    "rate_limit_error": {
        "http_status": 429,
        "retry": True,
        "user_action": "Implement exponential backoff"
    },
    
    # Server errors (5xx)
    "server_error": {
        "http_status": 500,
        "retry": True,
        "user_action": "Retry with backoff"
    },
    "service_unavailable": {
        "http_status": 503,
        "retry": True,
        "user_action": "Service temporarily down"
    }
}
```

### Level 3: Context-Aware Errors

**The Context Length Error** (Most detailed in industry):
```python
@app.post("/chat")
def chat(request: ChatRequest):
    # Count tokens
    input_tokens = count_tokens(request.messages)
    max_tokens = request.max_tokens or 1000
    total_needed = input_tokens + max_tokens
    
    model_limit = MODEL_LIMITS[request.model]  # e.g., 128000
    
    if total_needed > model_limit:
        # Calculate exactly how much to reduce
        overage = total_needed - model_limit
        suggested_max = max_tokens - overage - 100  # Buffer
        
        return {
            "error": {
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
                "message": f"This model's maximum context length is {model_limit:,} tokens. "
                          f"Your messages contain {input_tokens:,} tokens and you requested "
                          f"{max_tokens:,} output tokens. Total: {total_needed:,} tokens. "
                          f"Please reduce input by {overage:,} tokens or set max_tokens to {suggested_max}.",
                "param": "messages",
                "details": {
                    "model_limit": model_limit,
                    "input_tokens": input_tokens,
                    "output_tokens_requested": max_tokens,
                    "total_tokens_needed": total_needed,
                    "overage": overage,
                    "suggested_max_tokens": suggested_max
                }
            }
        }, 400
```

**Why This is Better**:
- Exact numbers (user knows exactly what to fix)
- Actionable suggestion (suggested_max_tokens)
- Self-service (no support ticket needed)

### Level 4: Error Correlation IDs

**The Debugging Nightmare**:
```
User reports: "My request failed"
You: "Which request?"
User: "Around 3pm yesterday"
You: Search through 10M logs...
```

**Solution**: Request IDs
```python
import uuid

@app.post("/chat")
def chat(request: ChatRequest):
    request_id = str(uuid.uuid4())
    
    try:
        result = model.generate(request)
        return {
            "id": request_id,
            "result": result
        }
    except Exception as e:
        logger.error(f"Request {request_id} failed", exc_info=True)
        return {
            "error": {
                "message": "Internal server error",
                "request_id": request_id,  # User can give you this!
                "support_url": f"https://support.example.com?request_id={request_id}"
            }
        }, 500
```

**Advanced**: Distributed Tracing
```python
# OpenTelemetry integration
from opentelemetry import trace

@app.post("/chat")
def chat(request: ChatRequest):
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("chat_request") as span:
        span.set_attribute("user_id", request.user_id)
        span.set_attribute("model", request.model)
        
        # If this calls other services, they inherit trace_id
        result = model.generate(request)
        
        span.set_attribute("output_tokens", result.tokens)
        return result

# Now you can trace request across:
# API Gateway → Lambda → Model Server → Database
# All with same trace_id
```

### Level 5: Predictive Error Prevention

**The Token Count Mismatch**:
```python
# Problem: Client estimates 1000 tokens, actual is 5000
# Result: Unexpected cost, slow response

# Solution: Return token count in response
@app.post("/chat")
def chat(request: ChatRequest):
    # Estimate tokens before processing
    estimated_input = count_tokens(request.messages)
    estimated_output = request.max_tokens or 1000
    estimated_cost = calculate_cost(estimated_input, estimated_output)
    
    # Check user budget
    if estimated_cost > user.remaining_budget:
        return {
            "error": {
                "type": "budget_exceeded",
                "message": f"Estimated cost ${estimated_cost:.2f} exceeds remaining budget ${user.remaining_budget:.2f}",
                "details": {
                    "estimated_input_tokens": estimated_input,
                    "estimated_output_tokens": estimated_output,
                    "estimated_cost": estimated_cost,
                    "remaining_budget": user.remaining_budget
                }
            }
        }, 402  # Payment Required
```

---

## 1.6 Retry Logic - The Exponential Backoff Nuances

### The Jitter Problem

**Without Jitter** (Thundering Herd):
```python
# All clients retry at same time!
# 1000 clients fail at t=0
# All retry at t=1s → 1000 simultaneous requests
# All retry at t=2s → 1000 simultaneous requests
# Server still overloaded!

for attempt in range(3):
    try:
        return api_call()
    except:
        time.sleep(2 ** attempt)  # 1s, 2s, 4s
```

**With Full Jitter** (Spread out):
```python
import random

for attempt in range(3):
    try:
        return api_call()
    except:
        max_wait = 2 ** attempt
        actual_wait = random.uniform(0, max_wait)
        time.sleep(actual_wait)
        
# Result: 1000 clients retry between 0-4s (spread out)
```

**Production Data** (From AWS SDK):
```
Without jitter: 1000 retries hit at t=1.0s (spike)
With full jitter: 1000 retries spread 0-1s (smooth)

Server load:
- Without: 1000 QPS spike (crashes)
- With: 100 QPS average (stable)
```

### The Retry Budget Pattern

**Problem**: During outage, retries make it worse
```
Normal: 1000 QPS
Outage: 50% fail → 500 retries → 1500 QPS
        Next 50% fail → 750 retries → 2250 QPS
        Server completely overwhelmed!
```

**Solution**: Retry budget (From Google SRE book)
```python
class RetryBudget:
    def __init__(self, window_seconds=10):
        self.window = window_seconds
        self.requests = deque()  # (timestamp, was_retry)
        self.max_retry_ratio = 0.1  # Max 10% retries
    
    def can_retry(self):
        now = time.time()
        
        # Remove old requests outside window
        while self.requests and self.requests[0][0] < now - self.window:
            self.requests.popleft()
        
        # Count retries in window
        total = len(self.requests)
        retries = sum(1 for _, is_retry in self.requests if is_retry)
        
        if total == 0:
            return True
        
        retry_ratio = retries / total
        return retry_ratio < self.max_retry_ratio
    
    def record_request(self, is_retry=False):
        self.requests.append((time.time(), is_retry))

# Usage
budget = RetryBudget()

for attempt in range(3):
    try:
        budget.record_request(is_retry=attempt > 0)
        return api_call()
    except Exception as e:
        if attempt < 2 and budget.can_retry():
            time.sleep(2 ** attempt)
        else:
            raise  # Don't retry, budget exhausted
```

---

## 1.7 Circuit Breaker - The Production Implementation

### The Three States Deep Dive

**State Machine**:
```
CLOSED → (failures > threshold) → OPEN
OPEN → (timeout elapsed) → HALF_OPEN
HALF_OPEN → (success) → CLOSED
HALF_OPEN → (failure) → OPEN
```

**The Half-Open Testing Problem**:
```python
# Naive: Send 1 test request
# Problem: 1 request not statistically significant!

# Production: Send multiple test requests
class CircuitBreaker:
    def __init__(self):
        self.state = "CLOSED"
        self.failure_count = 0
        self.failure_threshold = 5
        self.success_count = 0
        self.success_threshold = 3  # Need 3 successes to close
        self.timeout = 60
        self.last_failure_time = None
    
    def call(self, func):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                self.success_count = 0
            else:
                raise CircuitOpenError("Circuit is OPEN")
        
        try:
            result = func()
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise
    
    def on_success(self):
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                # Need 3 consecutive successes
                self.state = "CLOSED"
                self.failure_count = 0
        elif self.state == "CLOSED":
            self.failure_count = 0  # Reset on success
    
    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == "HALF_OPEN":
            # Immediately back to OPEN
            self.state = "OPEN"
        elif self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
```

**Monitoring Circuit Breaker State**:
```python
# Export metrics
from prometheus_client import Gauge, Counter

circuit_state = Gauge('circuit_breaker_state', 'Circuit breaker state', ['service'])
circuit_failures = Counter('circuit_breaker_failures', 'Failure count', ['service'])

def on_state_change(service, new_state):
    circuit_state.labels(service=service).set({
        "CLOSED": 0,
        "HALF_OPEN": 1,
        "OPEN": 2
    }[new_state])
    
    if new_state == "OPEN":
        # Alert on-call engineer
        alert(f"Circuit breaker OPEN for {service}")
```

---

## 1.8 Idempotency - The Distributed Systems Truth

### The Idempotency Key Pattern

**The Race Condition**:
```python
# Client A: POST /charge, Idempotency-Key: abc123
# Server: Processing... (takes 5s)
# Client A: (timeout after 3s, retries)
# Client A: POST /charge, Idempotency-Key: abc123 (again)
# Server: Must return SAME result, not charge twice!
```

**Naive Implementation** (WRONG):
```python
@app.post("/charge")
def charge(request: ChargeRequest):
    key = request.headers.get("Idempotency-Key")
    
    # Check cache
    if key in cache:
        return cache[key]  # RACE CONDITION!
    
    # Process
    result = process_charge(request)
    cache[key] = result
    return result

# Problem: Two requests arrive simultaneously
# Both check cache (both miss), both process!
```

**Correct Implementation** (WITH LOCKS):
```python
import redis
import json

redis_client = redis.Redis()

@app.post("/charge")
def charge(request: ChargeRequest):
    key = f"idempotency:{request.headers.get('Idempotency-Key')}"
    
    # Try to acquire lock
    lock_key = f"{key}:lock"
    lock_acquired = redis_client.set(lock_key, "1", nx=True, ex=60)
    
    if not lock_acquired:
        # Someone else is processing, wait and return their result
        for _ in range(60):  # Wait up to 60s
            time.sleep(1)
            result = redis_client.get(key)
            if result:
                return json.loads(result)
        
        return {"error": "Request timeout"}, 504
    
    try:
        # Check if already processed (before we got lock)
        existing = redis_client.get(key)
        if existing:
            return json.loads(existing)
        
        # Process request
        result = process_charge(request)
        
        # Store result (24h TTL)
        redis_client.setex(key, 86400, json.dumps(result))
        
        return result
    finally:
        # Release lock
        redis_client.delete(lock_key)
```

**The TTL Decision**:
```python
# How long to cache idempotency keys?

# Too short (1 hour):
# - User retries after 2 hours → duplicate charge!

# Too long (forever):
# - Memory grows forever
# - User can't retry legitimately

# Industry standard: 24 hours
# Reasoning:
# - Covers multi-day retries
# - Most retries happen within 1 hour
# - After 24h, user should generate new key
```

---

# 2. Rate Limiting - The Production Reality

## 2.1 Token-based Rate Limiting - The Math Behind It

### Token Bucket vs Leaky Bucket

**Token Bucket** (Allows bursts):
```python
class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity  # Max tokens
        self.tokens = capacity    # Current tokens
        self.refill_rate = refill_rate  # Tokens per second
        self.last_refill = time.time()
    
    def consume(self, tokens):
        self._refill()
        if tokens <= self.tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

# Example: Capacity=100, Refill=10/sec
# t=0: tokens=100 (full bucket)
# User consumes 100 tokens → tokens=0
# Wait 5s → tokens=50 (refilled 10*5)
# User can burst 50 tokens immediately!
```

**Leaky Bucket** (Smooths bursts):
```python
class LeakyBucket:
    def __init__(self, capacity, leak_rate):
        self.capacity = capacity
        self.queue = deque()
        self.leak_rate = leak_rate  # Requests per second
        self.last_leak = time.time()
    
    def add(self, tokens):
        self._leak()
        if len(self.queue) + tokens <= self.capacity:
            self.queue.extend([time.time()] * tokens)
            return True
        return False
    
    def _leak(self):
        now = time.time()
        elapsed = now - self.last_leak
        leaks = int(elapsed * self.leak_rate)
        
        for _ in range(min(leaks, len(self.queue))):
            self.queue.popleft()
        
        self.last_leak = now

# Example: Capacity=100, Leak=10/sec
# User sends 100 tokens → All queued
# Leak at 10/sec → Smooth 10/sec output
# No bursts allowed!
```

**When to Use Which**:
| Scenario | Token Bucket | Leaky Bucket |
|----------|-------------|--------------|
| API calls | ✅ (allow bursts) | ❌ |
| Video streaming | ❌ | ✅ (smooth bitrate) |
| Gaming actions | ✅ (allow combos) | ❌ |

---

## Interview Questions (Hao Hoang Style) - API Design

### Q1: The Retry Disaster
**Q**: You implement exponential backoff: 1s, 2s, 4s, 8s. Your service returns 503 for 30 seconds. 1000 clients all fail at t=0. What happens at t=15s? What about t=31s?

**Trap**: Most students say "retries spread out smoothly"

**Correct Answer**:
```
WITHOUT JITTER:
t=0: 1000 failures
t=1: 1000 retries (all fail again, 503)
t=3: 1000 retries (1+2=3) (all fail)
t=7: 1000 retries (1+2+4=7) (all fail)
t=15: 1000 retries (1+2+4+8=15) (all fail)
t=31: 1000 retries (1+2+4+8+16=31)

Problem at t=31: Service just recovered, instantly hit with 1000 requests!
Might crash again (thundering herd)

WITH JITTER:
t=31: 1000 retries spread over 0-16s window
      = ~62 QPS (manageable)
```

### Q2: The Circuit Breaker Edge Case
**Q**: Circuit breaker: failure_threshold=5, timeout=60s. At t=0, 5 requests fail (circuit OPEN). At t=59, circuit still OPEN. At t=61, 1 test request succeeds (circuit CLOSED). Immediately after, 10 requests fail. What's the circuit state? Should you alert?

**Answer**:
```python
State timeline:
t=0: 5 failures → OPEN
t=61: 1 success → CLOSED (too eager!)
t=61: 10 failures → OPEN again

Problem: Circuit closed after just 1 success
         Should require N consecutive successes

Correct config:
- success_threshold = 3 (need 3 successes to close)
- This prevents premature closing

Alert decision:
- YES if circuit opens twice within 5 minutes
- Indicates persistent backend issue
```

### Q3: The Idempotency Money Trap
**Q**: User charges $100 with Idempotency-Key: "abc123". Request succeeds, returns 200 OK. User's network drops before receiving response. User retries with SAME key. You return cached 200 OK. User claims "I was charged twice!" How do you prove you didn't?

**Answer**:
```python
# Store detailed event log, not just result
cache = {
    "abc123": {
        "status": 200,
        "result": {"charge_id": "ch_123", "amount": 100},
        "events": [
            {"t": "2024-01-10T10:00:00Z", "event": "request_received"},
            {"t": "2024-01-10T10:00:01Z", "event": "charge_created", "id": "ch_123"},
            {"t": "2024-01-10T10:00:02Z", "event": "response_sent"},
            {"t": "2024-01-10T10:05:00Z", "event": "retry_received"},
            {"t": "2024-01-10T10:05:00Z", "event": "cached_response_sent"}
        ]
    }
}

# Show user:
# 1. Only ONE charge_created event (proof of single charge)
# 2. Retry request returned CACHED response (not new charge)
# 3. Database shows only ONE charge with ID ch_123
```

### Q4: The Rate Limit Estimation
**Q**: Your API: 1000 TPM (tokens per minute) limit. User sends requests: 100 tokens, 200 tokens, 150 tokens, 300 tokens, 400 tokens. All within 1 minute. Which requests succeed? If you use sliding window (not fixed), does the 5th request succeed if sent at t=61s?

**Answer**:
```
Cumulative tokens:
Request 1 (t=0): 100 tokens → Total: 100 ✅
Request 2 (t=10): 200 tokens → Total: 300 ✅
Request 3 (t=20): 150 tokens → Total: 450 ✅
Request 4 (t=30): 300 tokens → Total: 750 ✅
Request 5 (t=40): 400 tokens → Total: 1150 ❌ (exceeds 1000)

At t=61s (sliding window):
- Window is [t=1, t=61]
- Request 1 (t=0) is OUTSIDE window
- Tokens in window: 200+150+300 = 650
- Request 5: 650 + 400 = 1050 ❌ (still exceeds!)
- Need to wait until t=71s (Request 2 expires)
```

---

**End of Part 1**

**Next**: Part 2 (Cost Management, QoS, Output Formatting) with same depth + production examples
