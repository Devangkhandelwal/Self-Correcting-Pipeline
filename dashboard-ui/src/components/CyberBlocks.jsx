import React from 'react';

// Cyber ASCII Matrix Art graphics inspired by Image 2
export const BrainMatrixBlock = () => {
  const asciiLines = [
    "      KK0&0#WS#0@&0#KU      ",
    "   ##WWXX0K#MW %WMW#W##   ",
    " MXX %SK%0W&#  %S%#0    ",
    " #0K    S#0K&BXW        ",
    "  #OK  %SW%0&%S#U  #OK  ",
    "   W0B  &#A&WEXB!0      ",
    "     B%WB  #0W&%S       ",
    "       0K%#S0S          "
  ];

  return (
    <div className="cyber-block-art">
      <pre className="cyber-ascii">{asciiLines.join('\n')}</pre>
    </div>
  );
};

export const ShieldMatrixBlock = () => {
  const asciiLines = [
    "      KK0&0#WS#0@&0#KU      ",
    "    ##WWXX0K#MW %WMW#W##    ",
    "   AMK0W1XBKB&XWW0B         ",
    "   SMMM1XK&BXB#         ",
    "   DRWOB &#A&WEXB!0        ",
    "   SMMM%W%#S!       ",
    "     %0S#MD         ",
    "       00           "
  ];

  return (
    <div className="cyber-block-art">
      <pre className="cyber-ascii">{asciiLines.join('\n')}</pre>
    </div>
  );
};

export const KeyMatrixBlock = () => {
  const asciiLines = [
    "      KK0&0#W               ",
    "      #0K&BXW#W%S0          ",
    "      MMW0K KW XM           ",
    "      WOB &#A&WEXBSOK       ",
    "      &#  #F&&WXBSOK       ",
    "      #0K %S#U              ",
    "          #0                ",
    "          #                 "
  ];

  return (
    <div className="cyber-block-art">
      <pre className="cyber-ascii">{asciiLines.join('\n')}</pre>
    </div>
  );
};
