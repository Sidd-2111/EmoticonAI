from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django import forms
from django.contrib.auth.models import User
from .models import JournalEntry, Profile

# myapp/forms.py
from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .models import Profile

class SignupForm(UserCreationForm):
    email = forms.EmailField(max_length=254, required=True, help_text='Required.')
    full_name = forms.CharField(max_length=255, required=False)
    phone_number = forms.CharField(max_length=20, required=False)
    
    class Meta(UserCreationForm.Meta):
        model = User
        fields = ('username', 'email')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add Bootstrap classes and placeholders
        self.fields['username'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Enter your username'
        })
        self.fields['email'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Enter your email'
        })
        self.fields['password1'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Enter your password'
        })
        self.fields['password2'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Confirm your password'
        })
        self.fields['full_name'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Enter your full name (optional)'
        })
        self.fields['phone_number'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Enter your phone number (optional)'
        })
    
    def save(self, commit=True):
        # The parent save() method will handle username, email, and a properly
        # hashed password. We just need to save the user, then the profile.
        user = super().save(commit=False)
        if commit:
            user.save()
            # Use update_or_create for robustness, in case a profile was
            # already created by a signal.
            Profile.objects.update_or_create(
                user=user,
                defaults={
                    'full_name': self.cleaned_data.get('full_name', ''),
                    'phone_number': self.cleaned_data.get('phone_number', '')
                }
            )
        return user


class JournalEntryForm(forms.ModelForm):
    content = forms.CharField(
        widget=forms.Textarea(
            attrs={
                'id': 'journalTextbox',
                'class': 'journal-textbox',
                'placeholder': "How are you feeling today? What's on your mind?",
                'rows': 7, # You can adjust the default number of rows
                'maxlength': '5000',
                'oninput': 'updateCharCounter()',
            }
        ),
        label="", # Set an empty label if the placeholder is sufficient
        help_text="Write your thoughts and feelings here."
    )
    class Meta:
        model = JournalEntry
        fields = ['content']


class UserUpdateForm(forms.ModelForm):
    # The 'username' field from Meta.fields will be used for the User's login username.
    # We can customize its label here if needed, or in Meta.labels.
    # The custom 'name' field is removed to avoid confusion with User.username.
    
    class Meta:
        model = User
        fields = ['username', 'email'] # Fields from the User model to include in the form
        labels = {
            'username': 'Username', # Explicit label for the User.username field
            'email': 'Email Address',
        }
        
    # The __init__ method is not strictly necessary here as ModelForm
    # will automatically initialize fields from the instance.

    def save(self, commit=True):
        # Call super().save(commit=False) to get the user instance
        # with 'username' and 'email' updated from cleaned_data, but not yet saved to DB.
        user = super().save(commit=False)

        # Now, derive first_name from the (potentially updated) username on the instance
        if user.username: # Ensure username is not empty
            username_parts = user.username.split(' ', 1)
            user.first_name = username_parts[0] if len(username_parts) > 0 else ''
            # Optionally, you could also set last_name:
            # user.last_name = username_parts[1] if len(username_parts) > 1 else ''
        
        if commit:
            user.save() # Save all changes to the database
        return user