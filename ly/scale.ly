\version "2.20.0"

\score {
  \relative c' { 
    \key d \major
    \repeat volta 2 { d4 e fis2 a4 b d2 }
  }
  \layout {}
  \midi { tempo = 60 }
}