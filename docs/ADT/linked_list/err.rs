fn error_prone() -> Result<i32, String> {
    Err("Returned an error".to_string())
}

fn always_ok() -> Result<i32, String> {
    Ok(0)
}

fn middle_function() -> Result<i32, String> {
    let ok_value = always_ok()?;
    let ok_value = error_prone()?;
    Ok(1)
}
