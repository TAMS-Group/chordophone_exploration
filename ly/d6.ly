\version "2.20.0"

\score {
  \unfoldRepeats {
  \relative d''' { 
    \repeat volta 2 { d4 d4 d2 }
  }
  }
  \layout {}
  \midi { tempo = 60 }
}
