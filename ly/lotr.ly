\version "2.20.0"

\score {
  \unfoldRepeats {
  \relative c' {
    \key d \major
    \repeat volta 2 { d4\pppp e fis2 a4 fis2 e4 d4 }
  }
  }
  %\layout {}
  \midi { tempo = 60 }
}
