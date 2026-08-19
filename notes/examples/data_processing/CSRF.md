# Cross Site Request Forgery

Source: https://cacm.acm.org/research/alloy/

Simulation of a cross-scripting browser attack, where malicious code acts like it is from a legitimate source in an already-authenticated browser session.

- one way to counter is to track origin of all server responses from the local browser

Example of a CSRF prevention system (that fails the constraint):

- could be useful as an example though, we don't really mind the bugs since there are dedicated model checkers for that - we just run the code

```Python
# Init Types/Sets
Endpoint
HTTPEvent
Client = Endpoint
Server = Endpoint
Response = HTTPEvent
Request = HTTPEvent
Redirect = Response

# Init relations
response: Request -> Response
embeds: Response -> Request

causes: Server -> HTTPEvent

from: HTTPEvent -> EndPoint
origin: HTTPEvent -> EndPoint
to: HTTPEvent -> EndPoint

# Constraints
forall r in Response: r.from in Client set
forall r in Response: r.from in Server set

# range of image must be associated with at least one request
forall r in Response: every response must have one request
# every response goes to the requesting location and vice versa
forall r in Response: r.to = r.response^-1.from and r.from = r.response^-1.to
# request cannot be embedded in a response to itself
forall r in Request: r not in closure(r.response.embeds)

# event can only be caused by a server if the event is from the server or a response from a server embeds an event
forall e in HTTPEvent, s in Server: e in s.causes iff e.from = s or exists r in Response: e in r.embeds and r in s.causes

# embedded requests within a response have the same origin as the response itself
forall r in Response, e in embeds[r]: e.origin = r.origin
# response origin either comes from a server directly or comes from a redirect with same origin as original request
forall r in Response: r.origin = response.r.origin if r in Redirect else r.from
# no embedded requests means its origin is the source (usually the browser)
forall r in Request: r.embeds = {} => r.origin in r.from

# Predicate
# Incoming requests allowed only if origin from this server or from a client
def enforceOrigins(s: Server):
    forall r in Requests: r.origin = r.to if r.to = s else r.origin = r.from

# Checks
!exists two servers, good and bad, such that:
    enforceOrigins(good) and
    # no requests from client reach the bad server
    !exists r in Request: r.to is bad and r.origin in Client and
    # some requests in the good server are bad
    exists r in Request: r.to is good and r in bad.causes
```
