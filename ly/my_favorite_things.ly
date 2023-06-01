\version "2.20.0"

\score {
  \unfoldRepeats {
  \relative c' {
    \key b \minor
    d4 a' a e d d a d d e d
    d4 a' a e d d a d d e d
  }
  }
  %\layout {}
  \midi { tempo = 120 }
}
