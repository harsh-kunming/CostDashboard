stock_bucket = {
    '0.50-0.69': (0.5,0.7),
    '0.70-0.89': (0.7,0.9),
    '0.90-0.99': (0.9,1),
    '1.00-1.25' : (1,1.26),
    '1.26-1.49' : (1.26,1.5),
    '1.50-1.99' : (1.5,2),
    '2.00-2.49' : (2,2.5),
    '2.50-2.99' : (2.5,3),
    '3.00-3.49' : (3,3.5),
    '3.50-3.99' : (3.5,4),
    '4.00-4.99' : (4,5),
    '5.00-7.99' : (5,8),
    '8.00-9.99' : (8,10),
    '10.00-14.99' : (10,15)
}


month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
              'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

color_map = {'Fancy Intense Yellow':'FIY',
             'Fancy Yellow':'FY',
             'Fancy Light Yellow':'FLY',
             'Fancy Vivid Yellow':'FVY',
             'W-X':'WXYZ',
             'Y-Z':'WXYZ'
}


def sort_buckets(bucket_list):
    """
    Sort bucket strings in ascending numerical order.

    Args:
        bucket_list: List of bucket strings (e.g., ['0.50-0.69', '10.00-14.99', '1.00-1.25'])

    Returns:
        List sorted by the starting numeric value of each bucket
    """
    def get_bucket_start_value(bucket_str):
        """Extract the starting numeric value from a bucket string."""
        try:
            # Extract the first number before the dash
            return float(bucket_str.split('-')[0])
        except (ValueError, IndexError):
            # If parsing fails, return a large number to sort it at the end
            return float('inf')

    return sorted(bucket_list, key=get_bucket_start_value)
