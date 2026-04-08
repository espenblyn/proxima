use std::borrow::Cow;

pub trait IntoSlice<'a, F: Clone> {
    fn into_slice(self) -> Cow<'a, [F]>;
}

impl<'a, F: Clone, const N: usize> IntoSlice<'a, F> for &'a [F; N] {
    #[inline]
    fn into_slice(self) -> Cow<'a, [F]> {
        Cow::Borrowed(self.as_slice())
    }
}

impl<'a, F: Clone> IntoSlice<'a, F> for &'a [F] {
    #[inline]
    fn into_slice(self) -> Cow<'a, [F]> {
        Cow::Borrowed(self)
    }
}

impl<'a, F: Clone> IntoSlice<'a, F> for &'a Vec<F> {
    #[inline]
    fn into_slice(self) -> Cow<'a, [F]> {
        Cow::Borrowed(self.as_slice())
    }
}

#[cfg(feature = "ndarray")]
use ndarray::{ArrayBase, ArrayView1, Data, Ix1};

#[cfg(feature = "ndarray")]
impl<'a, F, S> IntoSlice<'a, F> for &'a ArrayBase<S, Ix1>
where
    F: Clone,
    S: Data<Elem = F>,
{
    #[inline]
    fn into_slice(self) -> Cow<'a, [F]> {
        match self.as_slice() {
            Some(s) => Cow::Borrowed(s),
            None => {
                debug_assert!(
                    false,
                    "proxima-ml warning: Strided ndarray view caused a heap allocation!"
                );
                Cow::Owned(self.to_vec())
            }
        }
    }
}

#[cfg(feature = "ndarray")]
impl<'a, F> IntoSlice<'a, F> for ArrayView1<'a, F>
where
    F: Clone,
{
    #[inline]
    fn into_slice(self) -> Cow<'a, [F]> {
        match self.to_slice() {
            Some(s) => Cow::Borrowed(s),
            None => {
                debug_assert!(
                    false,
                    "proxima-ml warning: Strided ndarray view caused a heap allocation!"
                );
                Cow::Owned(self.to_vec())
            }
        }
    }
}
