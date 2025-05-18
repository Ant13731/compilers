import Relation.Binary.PropositionalEquality as Eq
open Eq using (_≡_; refl; sym; trans; cong)
open Eq.≡-Reasoning
open import Data.Nat.Properties using
  (+-assoc; +-identityˡ; +-identityʳ; *-assoc; *-identityˡ; *-identityʳ; *-distribʳ-+)

data LinkedList (A : Set) : Set where
  []  : LinkedList A
  _∷_ : A → LinkedList A → LinkedList A

infixr 5 _∷_

append : ∀ {A : Set} → LinkedList A → A → LinkedList A
append [] x = x ∷ []
append (y ∷ ys) x = y ∷ (append ys x)

-- concat
_++_ : ∀ {A : Set} → LinkedList A → LinkedList A → LinkedList A
[]       ++ ys  =  ys
(x ∷ xs) ++ ys  =  x ∷ (xs ++ ys)

-- Proof by induction that list concatenation is associative
++-assoc : ∀ {A : Set} (xs ys zs : LinkedList A)
  → (xs ++ ys) ++ zs ≡ xs ++ (ys ++ zs)
++-assoc [] ys zs =
  begin
    ([] ++ ys) ++ zs
  ≡⟨⟩
    ys ++ zs
  ≡⟨⟩
    [] ++ (ys ++ zs)
  ∎
++-assoc (x ∷ xs) ys zs =
  begin
    (x ∷ xs ++ ys) ++ zs
  ≡⟨⟩
    x ∷ (xs ++ ys) ++ zs
  ≡⟨⟩
    x ∷ ((xs ++ ys) ++ zs)
  ≡⟨ cong (x ∷_) (++-assoc xs ys zs) ⟩
    x ∷ (xs ++ (ys ++ zs))
  ≡⟨⟩
    x ∷ xs ++ (ys ++ zs)
  ∎
