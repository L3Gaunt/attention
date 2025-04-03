from flask import Flask, send_from_directory
from .attention import get_prompt_attention
import json
import os

app = Flask(__name__)

prompt = """
The 2008 Summer Olympics torch relay was run from March 24 until August 8, 2008, prior to the 2008 Summer Olympics, with the theme of "one world, one dream". Plans for the relay were announced on April 26, 2007, in Beijing, China. The relay, also called by the organizers as the "Journey of Harmony", lasted 129 days and carried the torch 137,000 km (85,000 mi) – the longest distance of any Olympic torch relay since the tradition was started ahead of the 1936 Summer Olympics.

After being lit at the birthplace of the Olympic Games in Olympia, Greece on March 24, the torch trav- eled to the Panathinaiko Stadium in Athens, and then to Beijing, arriving on March 31. From Beijing, the torch was following a route passing through six continents. The torch has visited cities along the Silk Road, symbolizing ancient links between China and the rest of the world. The relay also included an ascent with the flame to the top of Mount Everest on the border of Nepal and Tibet, China from the Chinese side, which was closed specially for the event.

Q: What was the theme
A:
""".strip()

prompt = """The genus of X is independent of the projective embedding, i. e. if X and Y are
isomorphic projective subschemes then g(X ) = g(Y ). See section 6.6.3 and exer-
cise 10.6.8 for more details.
(ii) If X is a smooth curve over C, then g(X ) is precisely the “topological genus”
introduced in example 0.1.1. (Compare for example the degree-genus formula of
example 0.1.3 with exercise 6.7.3 (ii).)
Remark 6.1.11. In general, the explicit computation of the Hilbert polynomial hX of a
projective subscheme X = Proj k[x0,...,xn]/I from the ideal I is quite complicated and
requires methods of computer algebra.
6.2. B ́ezout’s theorem. We will now prove the main property of the degree of a projective
variety: that it is “multiplicative when taking intersections”. We will prove this here only
for intersections with hypersurfaces, but there is a more general version about intersections
in arbitrary codimension (see e. g. cite Ha theorem 18.4).
Theorem 6.2.1. (B ́ezout’s theorem) Let X be a projective subscheme of Pn of positive
dimension, and let f ∈k[x0,...,xn] be a homogeneous polynomial such that no component
of X is contained in Z(f ). Then
deg(X ∩Z(f )) = deg X ·deg f .
Proof. The proof is very similar to that of the existence of the Hilbert polynomial in propo-
sition 6.1.5. Again we get an exact sequence
0 −→k[x0,...,xn]/I(X ) ·f−→k[x0,...,xn]/I(X ) −→k[x0,...,xn]/(I(X )+(f )) −→0
from which it follows that
χX ∩Z(f ) = χX (d)−χX (d −deg f ).
But we know that
χX (d) = deg X
m! dm +cm−1dm−1 +terms of order at most dm−2,
where m = dim X . Therefore it follows that
χX ∩Z(f ) = deg X
m! (dm −(d −deg f )m)+cm−1 (dm−1 −(d −deg f )m−1)
+terms of order at most dm−2
= deg X
m! ·m deg f ·dm−1 +terms of order at most dm−2.
We conclude that deg(X ∩Z(f )) = deg X ·deg f . 
Example 6.2.2. Let C1 and C2 be two curves in P2 without common irreducible com-
ponents. These curves are then given as the zero locus of homogeneous polynomials of
degrees d1 and d2, respectively. We conclude that deg(C1 ∩C2) = d1 ·d2 by B ́ezout’s the-
orem. By example 6.1.8 (i) this means that C1 and C2 intersect in exactly d1 ·d2 points, if
we count these points with their scheme-theoretic multiplicities in the intersection scheme
C1 ∩C2. In particular, as these multiplicities are always positive integers, it follows that C1
and C2 intersect set-theoretically in at most d1 ·d2 points, and in at least one point. This
special case of theorem 6.2.1 is also often called B ́ezout’s theorem in textbooks.
Example 6.2.3. In the previous example, the scheme-theoretic multiplicity of a point in
the intersection scheme C1 ∩C2 is often easy to read off from geometry: let P ∈C1 ∩C2 be
a point. Then:
(i) If C1 and C2 are smooth at P and have different tangent lines at P then P counts
with multiplicity 1 (we say: the intersection multiplicity of C1 and C2 at P is 1).
""".strip()

# Process only the prompt, no completion generation
result, tokenized, attn_m = get_prompt_attention(prompt)
sparse = attn_m.to_sparse()

@app.route("/attention")
def attention_view():
    indices, values = sparse.indices(), sparse.values()
    return json.dumps({
        'tokens': tokenized,
        'attn_indices': indices.T.numpy().tolist(),
        'attn_values': values.numpy().tolist(),
    })

@app.route("/")
def root():
    return send_from_directory('static', 'index.html')
