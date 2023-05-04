\version "2.20.0"

\score {
  \relative c' { 
    \key d \major
    \repeat volta 2 { d4 e fis2 a4 fis2 e d4 }
  }
  \layout {}
  \midi { tempo = 60 }
}