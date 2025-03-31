# File path
$filepath = Get-ChildItem -Path 'data\PPG_FieldStudy' -Filter *.zip -Recurse

# convert filepath to NameSpace object
$shell = new-object -com shell.application

# ForEach Loop processes each ZIP file located within the $filepath variable
foreach($file in $filepath)
{
    $zip = $shell.NameSpace($file.FullName)
    foreach($item in $zip.items())
    {
        $shell.Namespace($file.DirectoryName).copyhere($item)
    }
    # Remove-Item $file.FullName
}