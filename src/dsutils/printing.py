"""Printing.

"""

def print_table(header, data, 
                align=None,
                colsep='  ',
                newline='',
                latex=False,
                maxrows=None):
    """Pretty-print a table.
    
    Parameters
    ----------
    header : list of str
        List of headers for each column
    data : list of lists
        Data to print.  Should be a list of length Ncolumns, and each element
        a list of length Nrows.
    align : list of char
        Alignment for each column ('l', 'r', or 'c').  Default is to use left
        alignment for all columns.
    colsep : str
        Separator string between columns.  Default = '  '
    newline : str
        New line character.  Default = ''
    latex : bool
        Whether to print in LaTeX format.  Default = False.
    maxrows : int
        Maximum number of rows to print.  Default = None (print all rows)
    """
    
    # Check inputs
    if not isinstance(header, list):
        raise TypeError('header must be a list of str')
    if not all(isinstance(e, str) for e in header):
        raise TypeError('header must be a list of str')
    Nc = len(header)
    if not isinstance(data, list):
        raise TypeError('data must be a list of lists')
    if not all(isinstance(e, list) for e in data):
        raise TypeError('data must be a list of lists')
    if len(data) != Nc:
        raise ValueError('data must be same length as header')
    if any(len(e) != len(data[0]) for e in data):
        raise ValueError('all entries in data must be same length')
    if align is not None:
        if not isinstance(align, list):
            raise ValueError('align must be list of char')
        if not all(isinstance(e, str) for e in align):
            raise ValueError('align must be list of char')
        if len(align) != Nc:
            raise ValueError('align must be same length as header')
    if not isinstance(colsep, str):
        raise ValueError('colsep must be a str')
    if not isinstance(newline, str):
        raise ValueError('newline must be a str')
    if not isinstance(latex, bool):
        raise ValueError('latex must be True or False')
    if maxrows is not None and not isinstance(maxrows, int):
        raise ValueError('maxrows must be None or an int')

    # Length for each column
    lens = Nc*[None]
    for i in range(Nc):
        lens[i] = max([len(e) for e in [str(e2) for e2 in data[i]]])
        lens[i] = max(lens[i], len(header[i]))
        
    # Alignment
    if align is None:
        align = Nc*['l']
    
    # Latex colsep and newline
    if latex:
        colsep = ' & '
        newline = ' \\\\'
        print('\\begin{tabular}{ '+' '.join(e for e in align)+' }')
        print('\\hline')

    # Convert to format chars
    for i in range(Nc):
        if align[i]=='l':
            align[i] = '<'
        elif align[i]=='c':
            align[i] = '^'
        elif align[i]=='r':
            align[i] = '>'

    # Create the format string
    fmt_str = ""
    for i in range(Nc-1):
        fmt_str = fmt_str + "{:"+align[i]+str(lens[i])+"}"+colsep
    fmt_str = fmt_str + "{:"+align[Nc-1]+str(lens[Nc-1])+"}"+newline
        
    # Print header
    print(fmt_str.format(*header))

    # Latex header line
    if latex:
        print('\\hline')

    # Print the data
    if maxrows is None:
        maxrows = len(data[0])
    for r in range(maxrows):
        print(fmt_str.format(*[e[r] for e in data]))

    # Latex end
    if latex:
        print('\\hline')
        print('\\end{tabular}')


def describe_df(df):
    """Describe a DataFrame and its columns"""

    # Print all unique values if less than this many unique values
    max_unique = 10

    # Print number of rows and columns
    print('Rows:   ', df.shape[0])
    print('Columns:', df.shape[1])
    print('Memory usage:', df.memory_usage().sum(), 'Bytes')

    # Collect info about each column
    cols = []
    dtypes = []
    nulls = []
    mins = []
    means = []
    maxes = []
    modes = []
    u_strs = []
    for col in df:
        cols.append(col) 
        dtypes.append(str(df[col].dtype))
        nulls.append(df[col].isnull().sum())
        mins.append(df[col].min())
        if str(df[col].dtype) == 'object':
            means.append(' ')
        else:
            means.append(df[col].mean())
        maxes.append(df[col].max())
        t_mode = df[col].mode()
        if len(t_mode) == 0:
            modes.append(' ')
        else:
            modes.append(t_mode[0])
        n_unique = df[col].nunique()
        if n_unique >= max_unique:
            u_strs.append(str(n_unique)+' unique values')
        else:
            u_strs.append(str(df[col].unique()))

    # Print the per-column info
    print_table(
        ['Column', 'Dtype', 'Nulls', 'Min', 'Mean', 'Max', 'Mode', 'Uniques'],
        [cols, dtypes, nulls, mins, means, maxes, modes, u_strs]
    )
